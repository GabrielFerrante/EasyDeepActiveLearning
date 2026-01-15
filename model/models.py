import torch
import torch.nn as nn
import torch.nn.functional as F

class SVHNCustomCNN(nn.Module):
    def __init__(self, num_positions=5, num_classes=11):
        super(SVHNCustomCNN, self).__init__()
        
        # Bloco 1: Entrada 640x640 -> Saída 320x320
        self.layer1 = self._make_block(3, 32)
        # Bloco 2: 320x320 -> 160x160
        self.layer2 = self._make_block(32, 64)
        # Bloco 3: 160x160 -> 80x80
        self.layer3 = self._make_block(64, 128)
        # Bloco 4: 80x80 -> 40x40
        self.layer4 = self._make_block(128, 256)
        # Bloco 5: 40x40 -> 20x20
        self.layer5 = self._make_block(256, 512)
        # Bloco 6: 20x20 -> 10x10
        self.layer6 = self._make_block(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Cabeças de saída (uma para cada posição do dígito)
        # A entrada será 512 (saída do pooling)
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            ) for _ in range(num_positions)
        ])

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        logits = [head(features) for head in self.digit_heads]
        return torch.stack(logits, dim=1) # [Batch, 5, 11]