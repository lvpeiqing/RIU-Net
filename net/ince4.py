import torch
import torch.nn as nn


#### Inception Redidual Net

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        #
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4))
        # nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride,  padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4))
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1, c2, c3, c4], 1)

        # adding the skip connection
        out += residual
        out = self.relu(out)

        return out


class SElayer(nn.Module):
    def __init__(self, inplanes, ratio=0.25):
        super(SElayer, self).__init__()
        hidden_dim = int(inplanes * ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, inplanes, bias=False)
        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)

        return out


class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.se_layer = SElayer(out_ch)

    def forward(self, x):
        x_short = nn.Identity()(x)
        x = self.conv(x)
        x_1 = x + x_short
        x_2 = self.se_layer(x_1)
        x_3 = x + x_2
        return x_3


class Incrunet4(nn.Module):
    def __init__(self, in_ch, out_ch, args):
        super(Incrunet4, self).__init__()
        self.args = args

        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.e11 = IncResBlock(32, 32)
        self.e11_1 = ResidualConv(32, 32)

        self.pool1 = nn.MaxPool2d(2)

        self.e22 = IncResBlock(32, 64)
        self.e22_2 = ResidualConv(64, 64)

        self.pool2 = nn.MaxPool2d(2)

        self.e33 = IncResBlock(64, 128)
        self.e33_3 = ResidualConv(128, 128)

        self.pool3 = nn.MaxPool2d(2)

        self.e44 = IncResBlock(128, 256)
        self.e44_4 = ResidualConv(256, 256)


        self.pool4 = nn.MaxPool2d(2)

        # self.bridge = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )
        self.bridge = IncResBlock(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.d11 = IncResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.d22 = IncResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.d33 = IncResBlock(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.d44 = IncResBlock(64, 32)

        self.out = nn.Conv2d(32, out_ch, 1)


    def forward(self, x):
        e1 = self.e1(x)
        e11 = self.e11(e1)
        e11_1 = self.e11_1(e11)
        pool1 = self.pool1(e11_1)

        e22 = self.e22(pool1)
        e22_2 = self.e22_2(e22)
        pool2 = self.pool2(e22_2)

        e33 = self.e33(pool2)
        e33_3 = self.e33_3(e33)
        pool3 = self.pool3(e33_3)

        e44 = self.e44(pool3)
        e44_4 = self.e44_4(e44)
        pool4 = self.pool4(e44_4)

        bridge = self.bridge(pool4)

        up1 = self.up1(bridge)
        merge1 = torch.cat([up1,e44_4], dim=1)
        d11 = self.d11(merge1)

        up2 = self.up2(d11)
        merge2 = torch.cat([up2,e33_3], dim=1)
        d22 = self.d22(merge2)

        up3 = self.up3(d22)
        merge3 = torch.cat([up3,e22_2], dim=1)
        d33 = self.d33(merge3)

        up4 = self.up4(d33)
        merge4 = torch.cat([up4,e11_1], dim=1)
        d44 = self.d44(merge4)

        out = self.out(d44)
        # out = nn.Sigmoid()(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    # x = torch.randn(2,1,512,512).cuda()
    # in_shape = (1,512,512)
    unet = Incrunet4(in_ch=1, out_ch=2).cuda()

    summary(unet, (1, 512, 512)) #Total params:  5,267,810
    # y = unet(x)
    # print(y.shape) #torch.Size([4, 2, 512, 512])






