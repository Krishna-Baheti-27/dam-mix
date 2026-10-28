"""ResNet-32 for CIFAR (He et al., 2016) + heads used by all methods."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class ResNet32(nn.Module):
    """3 stages x 5 blocks, widths 16/32/64 -> 64-d feature. ~0.46M parameters."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for b in range(5):
                layers.append(BasicBlock(cin, cout, stride if b == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.out_dim = 64

    def forward(self, x):
        x = self.layers(self.stem(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class Net(nn.Module):
    """backbone f -> h (64-d)
       classifier: logits = W h + b
       projector g: z = normalize(g(h)) on the unit hypersphere (128-d)
       T: maps classifier weights to the projection space -> class prototypes p_c = normalize(T w_c)
          (the ECL coupling P = T W^T)."""

    def __init__(self, num_classes, proj_dim=128):
        super().__init__()
        self.backbone = ResNet32()
        d = self.backbone.out_dim
        self.classifier = nn.Linear(d, num_classes)
        self.projector = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, proj_dim))
        self.T = nn.Linear(d, proj_dim, bias=False)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.classifier(h)
        z = F.normalize(self.projector(h).float(), dim=1)
        return h, logits, z

    def prototypes(self):
        return F.normalize(self.T(self.classifier.weight).float(), dim=1)  # [C, proj_dim]
