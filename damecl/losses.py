"""All losses: CE family, SupCon, ECL and DAM-ECL (v1 = original blueprint, v2 = with rare-class fixes).

Notation follows the project summary:
  z_i  : L2-normalised embedding on the unit hypersphere
  p_c  : class prototype (p_c = normalize(T w_c), the ECL coupling)
  R_c  : rarity            = log(N_max / N_c)
  D_c  : directional dispersion = 1 - mean resultant length (== mean(1 - z_i . mu_c))
  m_c  : adaptive angular margin
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================ CE family
def ce_loss(logits, y):
    return F.cross_entropy(logits, y)


def weighted_ce_loss(logits, y, class_weights):
    return F.cross_entropy(logits, y, weight=class_weights)


def logit_adjusted_loss(logits, y, log_prior, tau=1.0):
    """Menon et al. 2021: add tau*log(prior) to the logits during training only."""
    return F.cross_entropy(logits + tau * log_prior, y)


def alc_loss(logits, y, log_prior, margins, scale):
    """Adaptive Logit Compensation = prior compensation (+log pi_c) and a class-specific
    margin subtracted from the TRUE class logit (in logit units: scale * m_y)."""
    adj = logits + log_prior
    adj = adj - scale * margins[y].unsqueeze(1) * F.one_hot(y, logits.shape[1]).float()
    return F.cross_entropy(adj, y)


def mixed_la_loss(logits, y_bg, y_fg, lam_bg, log_prior):
    """Logit-adjusted cross-entropy for a CutMix image: lam_bg of the loss for the background label,
    (1 - lam_bg) for the pasted (foreground) label."""
    logp = F.log_softmax(logits + log_prior, dim=1)
    return -(lam_bg * logp.gather(1, y_bg[:, None]).squeeze(1)
             + (1 - lam_bg) * logp.gather(1, y_fg[:, None]).squeeze(1)).mean()


def posthoc_tau(logits_probe, y_probe, log_prior, grid=None):
    """Post-hoc logit adjustment (Menon et al., 2021): predict argmax(logits - tau * log prior).
    tau is chosen to maximise balanced accuracy on the held-out probe (ties -> smallest |tau|)."""
    grid = np.round(np.arange(-1.0, 2.01, 0.1), 2) if grid is None else grid
    C = logits_probe.shape[1]
    best = (-1.0, 0.0)
    for t in sorted(grid, key=abs):
        pred = (logits_probe - t * log_prior).argmax(1)
        bacc = torch.stack([(pred[y_probe == c] == c).float().mean() for c in range(C)]).mean().item()
        if bacc > best[0] + 1e-9:
            best = (bacc, float(t))
    return best[1] + 0.0, best[0]


class GapTracker:
    """OUR SIGNAL. After every epoch: dispersion of each class on its (clean) training images vs on the
    held-out probe. Memorisation gap G_c = D_probe / D_train (>= 1 means the class looks tighter on
    training data than it really is). Smoothed across epochs."""

    def __init__(self, C, device, gamma=0.9, clip=6.0):
        self.C, self.gamma, self.clip = C, gamma, clip
        self.G = torch.ones(C, device=device)
        self.started = False
        self.history = []

    @torch.no_grad()
    def update(self, epoch, h_train, y_train, h_probe, y_probe):
        def disp(hh, yy):
            S = torch.zeros(self.C, hh.shape[1], device=hh.device).index_add_(0, yy, hh.float())
            n = torch.bincount(yy, minlength=self.C).float()
            return dispersion_from_sum(S, n)
        Dtr, Dpr = disp(h_train, y_train), disp(h_probe, y_probe)
        G_new = (Dpr / Dtr.clamp(min=1e-3)).clamp(1.0, self.clip)
        self.G = G_new if not self.started else self.gamma * self.G + (1 - self.gamma) * G_new
        self.started = True
        self.history.append({"epoch": epoch + 1, "D_train": Dtr.tolist(), "D_probe": Dpr.tolist(),
                             "G": self.G.tolist()})


def paste_probabilities(counts, mode, G=None, rarity_power=1.0, gap_power=1.0):
    """Which class gets pasted (as the CutMix foreground) how often.
    inv     : CMO default, q_c ~ 1 / N_c
    sqrtinv : q_c ~ 1 / sqrt(N_c)                       (ablation: rarity only, softer)
    gap     : q_c ~ N_c^(-rarity_power) * G_c^gap_power (OURS: rarity x measured memorisation gap)"""
    n = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1)
    if mode == "inv":
        q = 1.0 / n
    elif mode == "sqrtinv":
        q = n.pow(-0.5)
    elif mode == "gaponly":   # no class counts at all: only the measured memorisation gap
        q = (torch.ones_like(n) if G is None else G.cpu().float()).pow(gap_power)
    else:
        q = n.pow(-rarity_power) * (torch.ones_like(n) if G is None else G.cpu().float()).pow(gap_power)
    return q / q.sum()


# ============================================================================ contrastive
def supcon_loss(z, y, tau=0.1):
    """Supervised contrastive loss (Khosla et al., 2020). z: [n,d] normalised, y: [n]."""
    n = z.shape[0]
    sim = (z @ z.t()) / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    ok = n_pos > 0  # anchors without any positive are skipped
    loss = -(log_prob * pos).sum(1)[ok] / n_pos[ok]
    return loss.mean()


def psi(s, m):
    """Angular margin: cos(arccos(s) + m), stable form, with the ArcFace fallback when
    arccos(s) + m would exceed pi (keeps the function monotone)."""
    s = s.clamp(-1 + 1e-6, 1 - 1e-6)
    out = s * torch.cos(m) - torch.sqrt(1 - s * s) * torch.sin(m)
    th = torch.cos(math.pi - m)
    return torch.where(s > th, out, s - m * torch.sin(m))


def ecl_contrastive_loss(z, y, protos, num_classes, tau=0.1, margins=None):
    """Balanced class-wise contrastive loss of ECL (BC-ECL); with `margins` it becomes the
    DAM-ECL contrastive loss: similarities to the anchor's OWN class (positive pairs and own
    prototype, also where they appear in the denominator) are replaced by psi(s, m_{y_i}).

    For anchor i and positive j:
      loss = log( sum_c 1/|B_c| sum_{k in B_c, k!=i} exp(s_ik/tau) + sum_c exp(s_{i,p_c}/tau) )
             - (s_ij + s_{i,p_{y_i}}) / (2 tau)
    The 1/|B_c| averaging is what stops large classes dominating the denominator.
    """
    n = z.shape[0]
    C = num_classes
    S = z @ z.t()                       # [n,n] cosine similarities
    Sp = z @ protos.t()                 # [n,C] similarity to prototypes
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (y[:, None] == y[None, :]) & ~eye
    onehot = F.one_hot(y, C).float()    # [n,C]
    if margins is not None:
        m_row = margins[y]                                   # margin of the anchor's class
        S = torch.where(same, psi(S, m_row[:, None].expand(n, n)), S)
        own_p = psi(Sp.gather(1, y[:, None]), m_row[:, None])
        Sp = Sp.scatter(1, y[:, None], own_p)
    L = (S / tau).masked_fill(eye, -1e9)
    Lp = Sp / tau
    shift = torch.maximum(L.max(1).values, Lp.max(1).values).detach()
    expL = torch.exp(L - shift[:, None])
    per_class = expL @ onehot                                # sum over k in class c
    cnt = onehot.sum(0)[None, :] - onehot                    # |B_c| excluding the anchor itself
    per_class = torch.where(cnt > 0, per_class / cnt.clamp(min=1), torch.zeros_like(per_class))
    denom = per_class.sum(1) + torch.exp(Lp - shift[:, None]).sum(1)
    log_denom = torch.log(denom) + shift
    own_proto = Sp.gather(1, y[:, None]).squeeze(1)
    num = (S + own_proto[:, None]) / (2 * tau)               # [n,n], only used where `same`
    n_pos = same.sum(1)
    ok = n_pos > 0
    per_anchor = ((log_denom[:, None] - num) * same).sum(1)[ok] / n_pos[ok]
    return per_anchor.mean()


def alignment_loss(W, centroids, margins=None, use_sqrt_m=False):
    """Classifier <-> class-centroid alignment.
    margins=None   -> ECL's CC-GE:  || W/||W||_F - M/||M||_F ||_F^2
    margins given  -> MACA: rows weighted by Lambda = diag(sqrt(m_c / mean m)) (v2)
                      or diag(sqrt(m_c)) (v1, as written in the blueprint).
    W: [C,d] classifier weights, centroids: [C,d] (detached class means of normalised features).
    """
    if margins is not None:
        lam = torch.sqrt(margins.clamp(min=1e-8)) if use_sqrt_m else torch.sqrt(margins / margins.mean())
        W = W * lam[:, None]
        centroids = centroids * lam[:, None]
    A = W / W.norm()
    B = centroids / centroids.norm().clamp(min=1e-8)
    return ((A - B) ** 2).sum()


# ============================================================================ DAM-ECL state
def dispersion_from_sum(S, n, naive=False):
    """Directional dispersion D_c = 1 - (mean resultant length), per class.
    S: [C,d] sum of unit vectors per class, n: [C] counts.
    naive:          1 - ||S|| / n                               (biased: small classes look tight)
    bias-corrected: 1 - sqrt((||S||^2 - n) / (n (n - 1)))      (unbiased estimate of the squared length)"""
    n = n.clamp(min=2)
    if naive:
        D = 1 - S.norm(dim=1) / n
    else:
        rho2 = (S.pow(2).sum(1) - n) / (n * (n - 1))
        D = 1 - rho2.clamp(0, 1).sqrt()
    return D.clamp(0, 2)


class MarginState:
    """Turns per-class rarity and dispersion into the angular margins m_c.

    version='v1'      : the original blueprint  m = m0 + alpha*R + beta*D  (raw R, naive D, no ramp)
    version='v2'      : with the small fixes (bias-corrected D, normalised R and D, warm-up ramp, clip)
    source='train'    : D measured on the TRAINING embeddings of the epoch (what existing margin
                        methods do). Rare classes are memorised, so they look artificially tight.
                        floor=True adds the 'rare-class floor' heuristic (DAM-ECL v2).
    source='heldout'  : OUR IDEA. D measured on a small held-out probe set (images never trained on),
                        so it reflects how spread out each class really is.
    mode: 'full' | 'fixed' | 'rarity' | 'dispersion'  (ablations)
    Both D_train and D_heldout are logged every epoch (when a probe is given) for the plots.
    """

    def __init__(self, counts, version="v2", mode="full", m0=0.05, alpha=0.15, beta=0.15,
                 m_max=0.4, gamma=0.9, floor=True, source="train", device="cpu", proj_dim=128,
                 total_epochs=100):
        self.counts = torch.tensor(counts, dtype=torch.float32, device=device).clamp(min=1)
        self.C = len(counts)
        self.version, self.mode, self.source = version, mode, source
        self.m0, self.alpha, self.beta, self.m_max, self.gamma = m0, alpha, beta, m_max, gamma
        self.floor = floor and source == "train"
        self.total_epochs = total_epochs
        self.R = torch.log(self.counts.max() / self.counts)
        self.D = torch.zeros(self.C, device=device)
        self.has_D = False
        self.frozen = False
        self.S = torch.zeros(self.C, proj_dim, device=device)
        self.n = torch.zeros(self.C, device=device)
        self.history = []  # per-epoch D_train, D_heldout, D used, m

    @torch.no_grad()
    def accumulate(self, z, y):
        self.S.index_add_(0, y, z.detach().float())
        self.n.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

    @torch.no_grad()
    def end_epoch(self, epoch, probe_z=None, probe_y=None):
        D_train = dispersion_from_sum(self.S, self.n, naive=(self.version == "v1"))
        D_ho = None
        if probe_z is not None:
            Sp = torch.zeros_like(self.S).index_add_(0, probe_y, probe_z.float())
            npb = torch.bincount(probe_y, minlength=self.C).float()
            D_ho = dispersion_from_sum(Sp, npb)
        D_new = D_ho if (self.source == "heldout" and D_ho is not None) else D_train
        if not self.frozen:
            self.D = D_new if not self.has_D else self.gamma * self.D + (1 - self.gamma) * D_new
            self.has_D = True
        self.S.zero_()
        self.n.zero_()
        self.history.append({"epoch": epoch + 1, "D_train": D_train.tolist(),
                             "D_heldout": None if D_ho is None else D_ho.tolist(),
                             "D_used": self.effective_D().tolist(), "m": self.margins(epoch + 1).tolist()})

    def effective_D(self):
        D = self.D.clone()
        if self.version == "v2" and self.floor and self.has_D:
            # DAM-ECL v2 heuristic: a rare class never gets a SMALLER dispersion than the median class.
            rare = self.counts < self.counts.median()
            D[rare] = torch.maximum(D[rare], D.median())
        return D

    def margins(self, epoch):
        alpha = 0.0 if self.mode in ("fixed", "dispersion") else self.alpha
        beta = 0.0 if self.mode in ("fixed", "rarity") else self.beta
        D = self.effective_D()
        if self.version == "v1":
            m = self.m0 + alpha * self.R + beta * D
            return m.clamp(1e-3, math.pi / 2)
        Rn = self.R / self.R.max() if self.R.max() > 0 else self.R
        Dn = D / D.max() if (self.has_D and D.max() > 0) else torch.zeros_like(D)
        ramp = 1.0
        if self.total_epochs:
            start, full = 0.1 * self.total_epochs, 0.3 * self.total_epochs
            ramp = min(max((epoch - start) / max(full - start, 1), 0.0), 1.0)
        m = self.m0 + ramp * (alpha * Rn + beta * Dn)
        return m.clamp(self.m0, self.m_max)
