import torch
import torch.nn as nn
import time

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding="same"),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.net(input)

conv_net = ConvNet()
n = 500

r_imgs = torch.randn((500, 1, 1000, 1000))

s = time.time()
for i in range(n):
    out = conv_net(r_imgs[i])
e = time.time()
fw_time = (e - s) / n
print(f"fw time = {fw_time}")
