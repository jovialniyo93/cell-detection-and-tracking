import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DoubleConv, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class ConvOut(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(ConvOut, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0,bias=True),
            #nn.Softmax()
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Unet, self).__init__()

        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1=DoubleConv(ch_in,32)
        self.conv2=DoubleConv(32,64)
        self.conv3=DoubleConv(64,128)
        self.conv4=DoubleConv(128,256)
        self.conv5=DoubleConv(256,512)

        self.up6=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.conv6=DoubleConv(512,256)
        self.up7=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.conv7=DoubleConv(256,128)
        self.up8=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv8=DoubleConv(128,64)
        self.up9=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.conv9=DoubleConv(64,32)
        self.conv10=ConvOut(32,ch_out)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.maxpool(c1)
        c2=self.conv2(p1)
        p2=self.maxpool(c2)
        c3=self.conv3(p2)
        p3=self.maxpool(c3)
        c4=self.conv4(p3)
        p4=self.maxpool(c4)
        c5=self.conv5(p4)

        up6=self.up6(c5)
        concat6=torch.cat([up6,c4],dim=1)
        c6=self.conv6(concat6)

        up7=self.up7(c6)
        concat7=torch.cat([up7,c3],dim=1)
        c7=self.conv7(concat7)

        up8=self.up8(c7)
        concat8=torch.cat([up8,c2],dim=1)
        c8=self.conv8(concat8)

        up9=self.up9(c8)
        concat9=torch.cat([up9,c1],dim=1)
        c9=self.conv9(concat9)

        result=self.conv10(c9)
        return result





