import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Reference: https://arxiv.org/pdf/2183.10718 -> 4.1.2 Implementation Details

def build_resnet18_with_in_channels(in_channels: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if pretrained:
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            elif in_channels == 3:
                new_conv.weight.copy_(old_conv.weight)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.conv1 = new_conv
    return model


class FFTPreprocessor(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        # BT.709 luminosity grayscale formula
        # we want to capture spatial freqiency pattern, so color pattern does not matter
        gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]

        # shift low frequencies to the center
        f = torch.fft.fft2(gray, dim=(-2, -1))
        f = torch.fft.fftshift(f, dim=(-2, -1))

        # magnitude spectrum: capture how strong each frequency is
        mag = torch.log(torch.abs(f) + self.eps)

        mag_mean = mag.mean(dim=(-2, -1), keepdim=True)
        mag_std = mag.std(dim=(-2, -1), keepdim=True) + self.eps
        mag = (mag - mag_mean) / mag_std

        # [B, 1, H, W]
        return mag   


class ResNet18RGB(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = build_resnet18_with_in_channels(in_channels=3, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet18FFT1C(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.fft = FFTPreprocessor()
        self.backbone = build_resnet18_with_in_channels(in_channels=1, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = self.fft(x)
        return self.backbone(x_fft)


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        backbone = build_resnet18_with_in_channels(in_channels=in_channels, pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = backbone.fc.in_features 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class ResNet18RealArtifactNet(nn.Module):
    """
    RGB branch = artifact / fake evidence -> fake_score
    FFT branch = realness evidence  -> real_score

    fake_logit = alpha * fake_score - beta * real_score
    """
    def __init__(self, pretrained: bool = True, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.fft = FFTPreprocessor()

        self.artifact_branch = ResNet18FeatureExtractor(pretrained=pretrained, in_channels=3)
        self.realness_branch = ResNet18FeatureExtractor(pretrained=pretrained, in_channels=1)

        self.artifact_norm = nn.LayerNorm(self.artifact_branch.out_dim)
        self.realness_norm = nn.LayerNorm(self.realness_branch.out_dim)

        self.artifact_score = nn.Sequential(
            nn.Linear(self.artifact_branch.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # more positive -> more fake evidence
            nn.Linear(hidden_dim, 1),  
        )

        self.realness_score = nn.Sequential(
            nn.Linear(self.realness_branch.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # more positive -> more real evidence
            nn.Linear(hidden_dim, 1),   
        )

        # learnable positive weights
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        artifact_feat = self.artifact_norm(self.artifact_branch(x))
        realness_feat = self.realness_norm(self.realness_branch(self.fft(x)))

        fake_score = self.artifact_score(artifact_feat)  
        real_score = self.realness_score(realness_feat)   

        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        fake_logit = alpha * fake_score - beta * real_score + self.bias
        logits = torch.cat([-fake_logit, fake_logit], dim=1)  
        return logits