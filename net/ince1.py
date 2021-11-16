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
            nn.Conv2d(planes // 4, planes // 4, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU())
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(),
            nn.Conv2d(planes // 4, planes // 4, kernel_size=7, stride=stride, padding=3, bias=False),
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
        # out += residual
        out = self.relu(out)

        return out

class Incrunet1(nn.Module):
    def __init__(self, in_ch, out_ch, args):
        super(Incrunet1, self).__init__()
        self.args = args

        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.e11 = IncResBlock(32, 32)

        self.pool1 = nn.MaxPool2d(2)

        self.e22 = IncResBlock(32, 64)

        self.pool2 = nn.MaxPool2d(2)

        self.e33 = IncResBlock(64, 128)

        self.pool3 = nn.MaxPool2d(2)

        self.e44 = IncResBlock(128, 256)

        self.pool4 = nn.MaxPool2d(2)

        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

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
        pool1 = self.pool1(e11)

        e22 = self.e22(pool1)
        pool2 = self.pool2(e22)

        e33 = self.e33(pool2)
        pool3 = self.pool3(e33)

        e44 = self.e44(pool3)
        pool4 = self.pool4(e44)

        bridge = self.bridge(pool4)

        up1 = self.up1(bridge)
        merge1 = torch.cat([up1, e44], dim=1)
        d11 = self.d11(merge1)

        up2 = self.up2(d11)
        merge2 = torch.cat([up2, e33], dim=1)
        d22 = self.d22(merge2)

        up3 = self.up3(d22)
        merge3 = torch.cat([up3, e22], dim=1)
        d33 = self.d33(merge3)

        up4 = self.up4(d33)
        merge4 = torch.cat([up4, e11], dim=1)
        d44 = self.d44(merge4)

        out = self.out(d44)
        # out = nn.Sigmoid()(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    # x = torch.randn(2,1,512,512).cuda()
    # in_shape = (1,512,512)
    unet = Incrunet1(in_ch=1, out_ch=2).cuda()

    summary(unet, (1, 512, 512)) #Total params:  5,581,890
    # y = unet(x)
    # print(y.shape) #torch.Size([4, 2, 512, 512])





