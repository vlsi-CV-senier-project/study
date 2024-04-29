import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
  
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class RepConv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
        #self.deploy = deploy

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        if self.training :
            id_out = 0 if self.bn is None else self.bn(x)
            return self.act(self.conv1(x) + self.conv2(x) + id_out)
        else :
            return self.forward_fuse(x)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

class RepConv_Gaussian(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=0, g=g, act=False)  # p를 (p-k//2)에서 0으로 변경하여 padding을 제거

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        if self.training :
            id_out = 0 if self.bn is None else self.bn(x)
            return self.act(self.conv1(x) + self.conv2(x) + id_out)
        else :
            return self.forward_fuse(x)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.bn:
            kernelid, biasid = self._fuse_bn_tensor(self.bn)
        else:
            kernelid, biasid = 0, 0
        kernel1x1_padded = self._pad_1x1_to_3x3_tensor_with_gaussian(kernel1x1)
        return kernel3x3 + kernel1x1_padded + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def gaussian_kernel(kernel_size=3, sigma=1):
        axis = np.arange(0, kernel_size) - (kernel_size - 1) / 2
        x, y = np.meshgrid(axis, axis)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return torch.from_numpy(kernel).float()

    def _pad_1x1_to_3x3_tensor_with_gaussian(self, kernel1x1):
        # 1x1 커널을 3x3으로 패딩하는 로직 수정...
        if kernel1x1 is None:
            return 0
        else:
            gaussian = RepConv_Gaussian.gaussian_kernel(3, 1).to(kernel1x1.device)
            # expand 함수 대신 repeat 함수를 사용하여 채널 차원을 맞추는 방식으로 수정할 수 있습니다.
            gaussian = gaussian.repeat(kernel1x1.size(0), kernel1x1.size(1), 1, 1)
            # 여기서 padding=1 대신에 직접 1x1 커널을 3x3으로 만드는 방식으로 수정합니다.
            padded_kernel = F.pad(kernel1x1, (1, 1, 1, 1), "constant", 0)
            return padded_kernel * gaussian

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

def autopad_for_dilate(k, p=None, d=1):
    """Calculate padding automatically."""
    if p is None:
        p = ((k - 1) * (d - 1) + k - 1) // 2
    return p

class RepConv_dilate(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
        self.conv3 = Conv(c1, c2, k, s, autopad_for_dilate(k, d=d), g, d=d, act=False)  # Use autopad for dilated convolution
        #self.deploy = deploy

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        if self.training :
            id_out = 0 if self.bn is None else self.bn(x)
            return self.act(self.conv1(x) + self.conv2(x) + self.conv3(x) + id_out)
        else :
            return self.forward_fuse(x)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kerneldil, biasdil = self._fuse_bn_tensor(self.conv3)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kerneldil + kernelid, bias3x3 + bias1x1 + biasdil + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        self.__delattr__("conv3")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

class Repvgg(nn.Module):
    def __init__(self, num_classes=10):
        super(Repvgg, self).__init__()
        self.features = nn.Sequential(
            RepConv(3, 64, bn=True),
            RepConv(64, 64, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv(64, 128, bn=True),
            RepConv(128, 128, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv(128, 256, bn=True),
            RepConv(256, 256, bn=True),
            RepConv(256, 256, bn=True),
            RepConv(256, 256, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv(256, 512, bn=True),
            RepConv(512, 512, bn=True),
            RepConv(512, 512, bn=True),
            RepConv(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv(512, 512, bn=True),
            RepConv(512, 512, bn=True),
            RepConv(512, 512, bn=True),
            RepConv(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Repvgg_Gaussian(nn.Module):
    def __init__(self, num_classes=10):
        super(Repvgg_Gaussian, self).__init__()
        self.features = nn.Sequential(
            RepConv_Gaussian(3, 64, bn=True),
            RepConv_Gaussian(64, 64, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_Gaussian(64, 128, bn=True),
            RepConv_Gaussian(128, 128, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_Gaussian(128, 256, bn=True),
            RepConv_Gaussian(256, 256, bn=True),
            RepConv_Gaussian(256, 256, bn=True),
            RepConv_Gaussian(256, 256, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_Gaussian(256, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_Gaussian(512, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            RepConv_Gaussian(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
class Repvgg_dilate(nn.Module):
    def __init__(self, num_classes=10):
        super(Repvgg_dilate, self).__init__()
        self.features = nn.Sequential(
            RepConv_dilate(3, 64, bn=True),
            RepConv_dilate(64, 64, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_dilate(64, 128, bn=True),
            RepConv_dilate(128, 128, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_dilate(128, 256, bn=True),
            RepConv_dilate(256, 256, bn=True),
            RepConv_dilate(256, 256, bn=True),
            RepConv_dilate(256, 256, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_dilate(256, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RepConv_dilate(512, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            RepConv_dilate(512, 512, bn=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
