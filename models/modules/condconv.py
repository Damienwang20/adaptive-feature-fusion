import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class route_func(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, c_in, num_experts):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Conv1d(1, num_experts, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), 1, -1)
        x = self.avgpool(self.fc(x)).squeeze(-1)
        x = self.sigmoid(x)
        return x


class CondConv1d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.route_fn = route_func(in_channels, num_experts)

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        routing_weight = self.route_fn(x)

        b, c_in, t = x.size()
        k, c_out, c_in, kt = self.weight.size()
        x = x.view(1, -1, t)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kt)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv1d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv1d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-1))
        return output

class MultiScaleCondConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 experts_per_kernel=1):
        super(MultiScaleCondConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # precess kernel list
        self.kernel_sizes = list(sorted(kernel_sizes*experts_per_kernel))
        self.num_experts = len(self.kernel_sizes)

        self.route_fn = route_func(self.in_channels, self.num_experts)
        self.routing_weight = None
        
        self.weight = nn.Parameter(
            torch.Tensor(self.num_experts, out_channels, in_channels // groups, kernel_sizes[-1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_experts, out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        weight_mask = self.create_mask()
        self.weight_mask = nn.Parameter(torch.from_numpy(weight_mask), requires_grad=False)
        
    def create_mask(self):
        masks = []
        max_k = self.kernel_sizes[-1]
        for k in self.kernel_sizes:
            mask = np.zeros((self.out_channels, self.in_channels, max_k))
            l = (max_k-k)//2
            mask[:,:, l:l+k] = 1
            masks.append(mask)
        return np.stack(masks).astype(np.float32)

    def forward(self, x):
        self.weight.data = self.weight_mask * self.weight       
        self.routing_weight = self.route_fn(x)
        
        b, c_in, t = x.size()
        k, c_out, c_in, kt = self.weight.size()
        x = x.view(1, -1, t)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(self.routing_weight, weight).view(-1, c_in, kt)
        if self.bias is not None:
            combined_bias = torch.mm(self.routing_weight, self.bias).view(-1)
            output = F.conv1d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv1d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-1))
        return output