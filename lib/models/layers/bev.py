import torch.nn as nn

class Res_Block(nn.Module):
    def __init__(self, inplanes, planes, layers=2):
        super(Res_Block, self).__init__()
        self.blocks = nn.ModuleList()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        for i in range(layers):
            self.blocks.append(self._make_layer(planes, planes))
        
        self.relu = nn.ReLU()
        self.layers = layers
        self.maxpool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = self.conv(x)
        for i in range(self.layers):
            x = x + self.blocks[i](x)
            x = self.relu(x)
            x = self.maxpool(x)
        return x
    
    def _make_layer(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
        )