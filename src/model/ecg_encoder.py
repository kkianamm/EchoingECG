import torch
import torch.nn as nn

from src.model.gpo import GPO, l2norm

BASE_RESNET101 = [3, 4, 23, 3]


def conv1d(ni, nf, ks=3, stride=1, bias=False):
    """Create a 1D convolutional layer."""
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)


class ResBlock1D(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = conv1d(ni, nf, stride=stride)
        self.bn1 = nn.BatchNorm1d(nf)
        self.conv2 = conv1d(nf, nf)
        self.bn2 = nn.BatchNorm1d(nf)
        self.shortcut = nn.Sequential()
        if stride != 1 or ni != nf:
            self.shortcut = nn.Sequential(conv1d(ni, nf, ks=1, stride=stride), nn.BatchNorm1d(nf))

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class XResNet1D(nn.Module):
    def __init__(
        self, layers, numleads=12, init_dim=64, factor_dim=[2, 4, 8, 16], **kwargs
    ) -> None:
        super().__init__()
        self.in_planes = init_dim
        self.conv1 = conv1d(numleads, init_dim, ks=7, stride=2)
        self.bn1 = nn.BatchNorm1d(init_dim)

        self.layer1 = self._make_layer(init_dim * factor_dim[0], layers[0], stride=1)
        self.layer2 = self._make_layer(init_dim * factor_dim[1], layers[1], stride=2)
        self.layer3 = self._make_layer(init_dim * factor_dim[2], layers[2], stride=2)
        self.layer4 = self._make_layer(init_dim * factor_dim[3], layers[3], stride=2)

    def _make_layer(self, planes, num_blocks, stride) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock1D(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def return_features(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        return self.return_features(x)


class ProbXResNet1D(XResNet1D):
    def __init__(
        self,
        layers,
        numleads=12,
        init_dim=64,
        factor_dim=[2, 4, 8, 16],
        gpo_dim=32,
        embed_size=512,
        var_norm=False,
        mean_norm=True,
        **kwargs,
    ) -> None:
        super().__init__(
            layers=layers,
            numleads=numleads,
            init_dim=init_dim,
            factor_dim=factor_dim,
            **kwargs,
        )

        self.linear = nn.Linear(init_dim * factor_dim[3], embed_size)
        self.gpool = GPO(gpo_dim, gpo_dim)

        self.std_linear = nn.Linear(init_dim * factor_dim[3], embed_size)
        self.std_gpool = GPO(gpo_dim, gpo_dim)
        self.var_norm = var_norm
        self.mean_norm = mean_norm
        self.embed_size = embed_size

    @property
    def get_embed_dim(self):
        return self.embed_size

    def cal_out(self, x):
        pooled_features = self.linear(x.transpose(1, 2)) #batch, l, dim
        length = torch.tensor([x.size(2)]).to(x.device)
        pooled_features, _ = self.gpool(pooled_features, length)
        if self.mean_norm:
            pooled_features = l2norm(pooled_features, dim=-1)
        std_cap_emb = self.std_linear(x.transpose(1, 2))

        std_pooled_features, _ = self.std_gpool(std_cap_emb, length)
        if self.var_norm:
            std_cap_emb = l2norm(std_cap_emb, dim=-1)
        return {"mean": pooled_features, "std": std_pooled_features}

    def forward(self, x):
        ecg_features = self.return_features(x) #batch, dim, l
        return self.cal_out(ecg_features)


def xresnet1d101(numleads=12, init_dim=64, factor_dim=[2, 4, 8, 16]):
    return XResNet1D(BASE_RESNET101, numleads=numleads, init_dim=init_dim, factor_dim=factor_dim)


def prob_xresnet1d101(
    numleads=12,
    init_dim=64,
    factor_dim=[2, 4, 8, 16],
    gpo_dim=32,
    embed_size=512,
    var_norm=False,
    mean_norm=True,
):
    return ProbXResNet1D(
        BASE_RESNET101,
        numleads=numleads,
        init_dim=init_dim,
        factor_dim=factor_dim,
        gpo_dim=gpo_dim,
        embed_size=embed_size,
        var_norm=var_norm,
        mean_norm=mean_norm,
    )
