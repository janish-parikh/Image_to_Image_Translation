from re import X
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self._c7s1_64 = self.c7s1_64(3, 64, 7, 1)
        self._c7s1_3 = self.c7s1_3(64, 3, 7, 1)
        self._d128 = self.d_k(64, 128, 3, 2)
        self._d256 = self.d_k(128, 256, 3, 2)
        self._r256 = self.r_k(256)
        self._u128 = self.u_k(256, 128, 3, 2)
        self._u64 = self.u_k(128, 64, 3, 2)
        
    '''
    Input layer, c7s1−64
    a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
    '''
  
    def c7s1_64(self, in_features, out_features, kernel, stride):
        model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(out_features, affine=False),
            nn.ReLU(inplace=True),
        )

        return model

    '''
    Output layer, c7s1−3
    a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
    '''
    def c7s1_3(self, in_features, out_features, kernel, stride):
        model= nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel,
                stride=stride,
                padding=0,
                bias=False,
            ),
        )
        return model

    ''' 
    Downsampling
    dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
    '''
    def d_k(self, in_features, out_features, kernel, stride):
        model = nn.Sequential(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU(True),
        )
        return model

    # R256
    def r_k(self, in_features):
        return ResidualBlock(in_features)

    ''' 
    Upsampling, uk  
    uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer 
    with k filters and stride 1/2

    u128,u256
    '''
    def u_k(self, in_features, out_features, kernel, stride):
        model = nn.Sequential(
            nn.ConvTranspose2d(
                in_features,
                out_features,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(out_features, affine=False),
            nn.ReLU(inplace=True),
        )
        return model

    def forward(self, x):
        #c7s1 − 64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1 − 3
        x = self._c7s1_64(x)
        x = self._d128(x)
        x = self._d256(x)
        
        # 9 ResidualBlocks
        for i in range(9):
            x = self._r256(x)
        x = self._u128(x)
        x = self._u64(x)
        x = self._c7s1_3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        Convolutions layers
        Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
        '''
        # C64
        self._c64 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # C128
        self._c128 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # C256
        self._c256 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # C512
        self._c512 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # FCN classification layer
        self._c1 = nn.Conv2d(512, 1, 4, stride=1, padding=1)

    def forward(self, x):
        #C64 − C128 − C256 − C512
        x = self._c64(x)
        x = self._c128(x)
        x = self._c256(x)
        x = self._c512(x)   
        x = self._c1(x)
        return x


class ResidualBlock(nn.Module):
    '''
    Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same
    number of filters on both layer.
    '''
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            )

    def forward(self, x):
        return x + self.conv_block(x)

