from __init__ import *
from utils import round_width


def Depthwise_Conv2d(
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1704.04861

    The structure is decribed in <Figure 2.(b)> of the paper.

    'padding' is set to retain 2D-size of a feature, if it is not given.
    """

    if not padding and padding != 0:
        padding = dilation * (kernel_size - 1) // 2

    dw_conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                        groups=channels, bias=bias)
    return dw_conv



def Pointwise_Conv2d(
        in_channels: int,
        out_channels: int,
        bias: bool = False):

    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1704.04861

    The structure is decribed in <Figure 2.(c)> of the paper.
    """

    pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
    return pw_conv



def Seperable_Conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = False):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022
    """

    conv = nn.Sequential(Depthwise_Conv2d(in_channels, kernel_size, stride, padding, dilation, False),
                         Pointwise_Conv2d(in_channels, out_channels, bias))
    return conv



class Mish(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1908.08681
    """

    def __init__(self,
                 beta: int = 1,
                 threshold: int = 20):

        super(Mish, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        return x * F.softplus(x, self.beta, self.threshold).tanh()



class H_Sigmoid(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace:bool = False):
        super(H_Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6



class H_Swish(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace:bool = False):
        super(H_Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6



class L2_Norm(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1205.2653
    """

    def __init__(self,
                 channels: int,
                 eps: float = 1e-10):

        super(L2_Norm,self).__init__()

        self.weight = nn.Parameter(torch.Tensor(channels))
        self.channels = channels
        self.eps = eps


    def forward(self, x):
        norm = x.pow(2).sum(1, True).sqrt() + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

        return x



class Static_ConvLayer(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    The module selectively comprises convolution, batch normalizaion, activation in general order.

    'padding', 'dilation', 'groups' of nn.Conv2d are given constantly, according to 'Static'.
    'Act' receives an instance, not a class.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = False,
                 batch_norm: bool = True,
                 Act: None or nn.Module = nn.ReLU(inplace=False),
                 **kwargs):

        batch_eps = kwargs.get('eps', 1e-05)
        batch_momentum = kwargs.get('momentum', 0.1)

        padding = (kernel_size - 1) // 2


        super(Static_ConvLayer, self).__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]

        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
        if Act:
            layer.append(Act)

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)



class Dynamic_ConvLayer(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    'reverse' orders components in various ways.
    'padding' is set to retain 2D-size of a feature, if it is not given.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 batch_norm: bool = True,
                 Act: None or nn.Module = nn.ReLU(inplace=False),
                 reverse: str = None,
                 **kwargs):

        batch_eps = kwargs.get('eps', 1e-05)
        batch_momentum = kwargs.get('momentum', 0.1)

        if not padding and padding != 0:
            padding = dilation * (kernel_size - 1) // 2


        super(Dynamic_ConvLayer, self).__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]

        if not reverse:
            if batch_norm:
                layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.append(Act)

        elif reverse == 'ACB':
            if batch_norm:
                layer.append(nn.BatchNorm2d(out_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(0, Act)

        elif reverse == 'BAC':
            if batch_norm:
                layer.insert(0, nn.BatchNorm2d(in_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(-1, Act)

        elif reverse == 'ABC':
            if batch_norm:
                layer.insert(0, nn.BatchNorm2d(in_channels, eps=batch_eps, momentum=batch_momentum))
            if Act:
                layer.insert(0, Act)

        else:
            raise ValueError('reverse order should be one of ACB, BAC, ABC')

        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)



class Squeeze_Excitation(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1709.01507

    The structure is decribed in <Figure 2.(right)> of the paper.

    'reduction' is a denominator of reduction ratio of squeezing, following the paper.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 reduction: float,
                 batch_norm: bool = False,
                 Act: nn.Module = nn.ReLU(),
                 Sigmoid: nn.Module = nn.Sigmoid(),
                 **kwargs):

        divisor = kwargs.get('divisor', 1)
        round_bias = kwargs.get('round_bias', 0.9)

        reduct_channels = round_width(channels // reduction, divisor, round_bias)


        super(Squeeze_Excitation, self).__init__()

        squeeze = [nn.AdaptiveAvgPool2d(1),
                   nn.Flatten(),
                   nn.Linear(in_channels, reduct_channels)]

        if not batch_norm:
            squeeze.append(Act)
        else:
            squeeze.append(nn.BatchNorm2d(reduct_channels))
            squeeze.append(Act)

        excitation = [nn.Linear(reduct_channels, channels),
                      Sigmoid]

        self.squeeze = nn.Sequential(*squeeze)
        self.excitation = nn.Sequential(*excitation)


    def forward(self, input):
        batch, channel, _, _ = input.size()

        x = self.squeeze(input)
        x = self.excitation(x)
        x = x.view(batch, channel, 1, 1)
        x = x * input
        return x



class Squeeze_Excitation_Conv(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    From Squeeze_Excitation, nn.Linear is replaced with nn.Conv2d with filter of 1.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 reduction: float,
                 batch_norm: bool = False,
                 Act: nn.Module = nn.ReLU(),
                 Sigmoid: nn.Module = nn.Sigmoid(),
                 **kwargs):

        divisor = kwargs.get('divisor', 1)
        round_bias = kwargs.get('round_bias', 0.9)

        reduct_channels = round_width(channels // reduction, divisor, round_bias)


        super(Squeeze_Excitation_Conv, self).__init__()

        squeeze = [nn.AdaptiveAvgPool2d(1),
                   nn.Conv2d(in_channels, reduct_channels, kernel_size=1)]

        if not batch_norm:
            squeeze.append(Act)
        else:
            squeeze.append(nn.BatchNorm2d(reduct_channels))
            squeeze.append(Act)

        excitation = [nn.Conv2d(reduct_channels, channels, kernel_size=1),
                      Sigmoid]

        self.squeeze = nn.Sequential(*squeeze)
        self.excitation = nn.Sequential(*excitation)


    def forward(self, input):
        x = self.squeeze(input)
        x = self.excitation(x)
        x = x * input
        return x



class SPP(nn.Module):
    """
    __version__ = 1.0
    __date__ = Mar 7, 2022

    paper : https://arxiv.org/abs/1406.4729

    Each pooling is followed by flattening to vector in the original, but here, it isn't.
    """

    def __init__(self,
                 kernel_sizes: list,
                 inverse: bool = False):

        super(SPP, self).__init__()

        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, padding=k // 2) for k in kernel_sizes])
        self.inverse = inverse


    def forward(self, x):
        x = [x] + [p(x) for p in self.pools]

        if self.inverse:
            x = x[::-1]

        x = torch.cat(x, 1)
        return x
