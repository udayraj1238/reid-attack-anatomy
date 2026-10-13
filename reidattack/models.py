"""ReID backbones with a BNNeck head (Bag-of-Tricks baseline).
Input: images in [0,1]. Mean/std normalisation happens INSIDE forward, so every attack
can perturb raw pixels and the epsilon budget means what it says (e.g. 8/255)."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision


class ReIDNet(nn.Module):
    def __init__(self, arch="resnet50", num_classes=751, pretrained=True):
        super().__init__()
        self.arch = arch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if arch == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet50(weights=w)
            m.layer4[0].conv2.stride = (1, 1)           # last-stride=1: finer feature map
            m.layer4[0].downsample[0].stride = (1, 1)
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                      m.layer1, m.layer2, m.layer3, m.layer4)
            self.pool, dim = nn.AdaptiveAvgPool2d(1), 2048
        elif arch == "densenet121":
            w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.densenet121(weights=w)
            self.body = nn.Sequential(m.features, nn.ReLU(inplace=True))
            self.pool, dim = nn.AdaptiveAvgPool2d(1), 1024
        elif arch == "vit_small":
            import timm
            self.body = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                                          pretrained=pretrained, img_size=(256, 128), num_classes=0)
            self.pool, dim = None, 384
        else:
            raise ValueError(arch)
        self.dim = dim
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def features(self, x):
        x = (x - self.mean) / self.std
        f = self.body(x)
        if self.pool is not None:
            f = self.pool(f).flatten(1)
        return f, self.bn(f)

    def embed(self, x):
        """L2-normalised retrieval embedding (what ReID is actually evaluated on)."""
        return F.normalize(self.features(x)[1], dim=1)

    def logits(self, x):
        return self.classifier(self.features(x)[1])

    def forward(self, x):  # training: (triplet feature, class logits)
        f, fb = self.features(x)
        return f, self.classifier(fb)
