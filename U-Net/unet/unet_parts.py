import torch
import torch.nn as nn
import torch.nn.functional as F


#从UNet网络中可以看出，不管是下采样过程还是上采样过程，每一层都会连续进行两次卷积操作，
#这种操作在UNet网络中重复很多次，可以单独写一个DoubleConv模块
class DoubleConv(nn.Module):
    #(convolution => [BN] => ReLU) * 2
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            #padding's default value = 0 in Conv2d of Pytorch
            #在U-Net论文中是无padding的，即padding = 0
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            #inplace = 1节省内存
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

#下采样模块，在U-Net中一共执行4次下采样，就是一个maxpool池化层，进行下采样，然后接一个DoubleConv模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            #max pooling operation with stride 2 for downsampling
            nn.MaxPool2d(2),
            #Down和DoubleConv类都是Module的子类，所以可以嵌套调用
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


#在上采样过程中进行Skip Connection和UpSample，下采样也是进行4次
#U-Net Skip Connection的融合操作：就是将feature map的通道进行叠加，俗称Concat
#Concat操作也很好理解，举个例子：一本大小为10cm*10cm，厚度为3cm的书A，和一本大小为10cm*10cm，厚度为4cm的书B。
#将书A和书B，边缘对齐地摞在一起。这样就得到了，大小为10cm*10cm厚度为7cm的一摞书。
#在实际使用中，Concat融合的两个feature map的大小不一定相同。
#例如256*256*64的feature map和240*240*32的feature map进行Concat。
#这种时候，就有两种办法：
#第一种：将大256*256*64的feature map进行裁剪，裁剪为240*240*64的feature map，
#比如上下左右，各舍弃8 pixel，裁剪后再进行Concat，得到240*240*96的feature map。
#第二种：将小240*240*32的feature map进行padding操作，padding为256*256*32的feature map。
#比如上下左右，各补8 pixel，padding后再进行Concat，得到256*256*96的feature map。
#UNet采用的Concat方案就是第二种，将小的feature map进行padding，padding的方式是补0，一种常规的常量填充。
class Up(nn.Module):
    #Upscaling then double conv
    #bilinear表示双线性插值
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        #scale_factor 指定高度、宽度扩大倍数，这里都扩大2倍,这里的通道数不会改变！！！
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        # input is CHW， x1接收的是上采样的数据，x2接收的是特征融合的数据（来自下采样）
        #First，Upsample
        x1 = self.up(x1)
        #Second，特征融合
        #feature map的size为batch_size * channles * height * width
        diffY = x2.size()[2] - x1.size()[2] #高度差
        diffX = x2.size()[3] - x1.size()[3] #宽度差
        #torch.nn.functional.pad是PyTorch内置的矩阵填充函数
        #F.pad的第2个参数如果为4元素tensor，指的是（左填充，右填充，上填充，下填充），其数值代表填充次数
        #如果第4个参数，不填写，默认为0，即用0填充
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #将x1拼接在x2后（对应论文中的图），dim = 1表示按照（列）拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)