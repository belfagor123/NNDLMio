import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):

    # Generic class for ResNet18 and ResNet34, bigger models modify the blocks
    # resnet18 = ResNet(ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
    # resnet34 = ResNet(ResidualBlock, [3, 4, 6, 3],num_classes=num_classes).to(device) 

    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ConvBlock (nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
        super(ConvBlock,self).__init__()
        
        self.conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        
        x=self.conv2d(x)
        x=self.batchnorm(x)
        x=self.relu(x)
        
        return x
class InceptionBlock (nn.Module):
    
    """
    int_nxn is the intermediate output dimension for the branch that does nxn convolution
    """
    
    def __init__(self , in_channels , out_1x1 , int_3x3 , out_3x3 , int_5x5 , out_5x5 , out_1x1_pooling):
        super(InceptionBlock,self).__init__()
        
        self.path1 = ConvBlock(in_channels=in_channels,out_channels=out_1x1,kernel_size=1,stride=1,padding=0)
        self.path2 = nn.Sequential(ConvBlock(in_channels=in_channels,out_channels=int_3x3,kernel_size=1,stride=1,padding=0),
                                   ConvBlock(in_channels=int_3x3,out_channels=out_3x3,kernel_size=3,stride=1,padding=1))    #padding is set to have same output dimension as input
        self.path3 = nn.Sequential(ConvBlock(in_channels=in_channels,out_channels=int_5x5,kernel_size=1,stride=1,padding=0),
                                   ConvBlock(in_channels=int_5x5,out_channels=out_5x5,kernel_size=5,stride=1,padding=2))    #padding is set to have same output dimension as input
        self.path4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                                   ConvBlock(in_channels=in_channels,out_channels=out_1x1_pooling,kernel_size=1,stride=1,padding=0))
        
    def forward(self,x):
        
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)
        
        tot = torch.cat([p1,p2,p3,p4],dim=1)
        
        return tot
    
class InceptionV1 (nn.Module):
    def __init__(self , in_channels , num_classes):
        super(InceptionV1,self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.conv2 =  nn.Sequential(ConvBlock(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0),
                                    ConvBlock(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling
        self.inception3a = InceptionBlock(in_channels=192,out_1x1=64,int_3x3=96,out_3x3=128,int_5x5=16,out_5x5=32,out_1x1_pooling=32)
        self.inception3b = InceptionBlock(in_channels=256,out_1x1=128,int_3x3=128,out_3x3=192,int_5x5=32,out_5x5=96,out_1x1_pooling=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionBlock(in_channels=480, out_1x1=192, int_3x3=96, out_3x3=208, int_5x5=16, out_5x5=48, out_1x1_pooling=64)
        self.inception4b = InceptionBlock(in_channels=512, out_1x1=160, int_3x3=112, out_3x3=224, int_5x5=24, out_5x5=64, out_1x1_pooling=64)
        self.inception4c = InceptionBlock(in_channels=512, out_1x1=128, int_3x3=128, out_3x3=256, int_5x5=24, out_5x5=64, out_1x1_pooling=64)
        self.inception4d = InceptionBlock(in_channels=512, out_1x1=112, int_3x3=144, out_3x3=288, int_5x5=32, out_5x5=64, out_1x1_pooling=64)
        self.inception4e = InceptionBlock(in_channels=528, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionBlock(in_channels=832, out_1x1=256, int_3x3=160, out_3x3=320, int_5x5=32, out_5x5=128, out_1x1_pooling=128)
        self.inception5b = InceptionBlock(in_channels=832, out_1x1=384, int_3x3=192, out_3x3=384, int_5x5=48, out_5x5=128, out_1x1_pooling=128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear( 1024 , num_classes)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
