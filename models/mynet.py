import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from base import BaseModel

from .modules import MultiScaleCondConv1d

activate_fn = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'p_relu': nn.PReLU(),
    'r_relu': nn.RReLU()
}

def conv_bn_activate(in_channel, out_channel, kernel_list, num_experts, padding, dilation=1, activate='relu'):
    if activate is not None:
        conv_bn = nn.Sequential(
            MultiScaleCondConv1d(in_channel, out_channel, kernel_list, experts_per_kernel=num_experts, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channel),
            activate_fn[activate])
    else:
        conv_bn = nn.Sequential(
            MultiScaleCondConv1d(out_channel, out_channel, kernel_list, experts_per_kernel=num_experts, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channel))
    return conv_bn

class MSConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_list, num_experts=3, padding=0, dilation=1, activate='relu'):
        super(MSConvResBlock, self).__init__()
        
        self.conv2 = conv_bn_activate(in_channel, out_channel, kernel_list, num_experts, padding, dilation, activate)
        self.conv3 = conv_bn_activate(out_channel, out_channel, kernel_list, num_experts, padding, dilation, None)
        
        self.act_fn = activate_fn[activate]

        self.downsample = None 
        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1),
                nn.BatchNorm1d(out_channel))

    def forward(self, x):
        residual = x
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = self.act_fn(x + residual)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MSCResNet(BaseModel):
    def __init__(self, seq_len, n_class, in_channel=1, out_channel=64, 
            kernel_list=[[],[],[]],
            num_experts=[],
            padding = [],
            dilation = [1,1,1],
            activate ='p_relu'):
        super().__init__()
        self.n_class = n_class
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block1 = nn.Sequential(
            MSConvResBlock(self.in_channel, self.out_channel, kernel_list[0], num_experts[0], padding[0], dilation[0], activate),
            nn.MaxPool1d(2),
            SELayer(self.out_channel),
            nn.Dropout(0.3))
        self.block2 = nn.Sequential(
            MSConvResBlock(self.out_channel, self.out_channel*2, kernel_list[1], num_experts[1], padding[1], dilation[1], activate),
            nn.MaxPool1d(2),
            SELayer(self.out_channel*2),
            nn.Dropout(0.3))
        self.block3 = nn.Sequential(
            MSConvResBlock(self.out_channel*2, self.out_channel*2, kernel_list[2], num_experts[2], padding[2], dilation[2], activate),
            nn.MaxPool1d(2),
            SELayer(self.out_channel*2),
            nn.Dropout(0.3))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.out_channel*2, self.n_class)
    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        p3 = self.gap(f3).squeeze(-1)
        feat = p3
        y = self.fc(feat)
        return y, feat

class EmbedNet(BaseModel):
    def __init__(self, n_class, N=1, in_dim=128, out_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_class = n_class

        self.N = N # N proxies for each class
        self.proxies = nn.Parameter(torch.Tensor(n_class*N, out_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim))

    def forward(self, feat, label=None):
        embed = self.embedding(feat)
        embed = F.normalize(embed, p=2, dim=1)                 # l2 norm
        proxies = F.normalize(self.proxies, p=2, dim=1)        # l2 norm

        ## generate dist feature
        dist_feat = self.generate_feature(embed, proxies)

        ## train proxy based on classify
        nc = self.n_class
        dist = euclidean_dist(embed, proxies)         # B, NxC
        similarity = dist.view(dist.size(0), nc, -1)  # B, C, N
        similarity = torch.mean(similarity, dim=-1)   # B, C
        similarity = torch.exp(-1*similarity)

        return similarity, dist_feat, (embed, proxies)
    
    def generate_feature(self, embed, proxies):
        dist_feat = elementwise_l2_dist(embed, proxies)     # B, N*C, D
        dist_feat = 3*F.normalize(dist_feat, dim=-1)
        return dist_feat

## 普通卷积
class ConvFuse(nn.Module):
    def __init__(self, channels=64, mode='cat'):
        """channels: number of class
           mode: ['add', 'cat'], way of fuse, add or cat.
        """
        super().__init__()
        self.mode = mode
        self.conv_cat = nn.Conv1d(channels+1, 1, 1)
        self.conv_add = nn.Conv1d(channels, 1, 1)
        
    def forward(self, cnn_f, dist_f):
        if self.mode == 'cat':
            out = self.conv_cat(torch.cat([cnn_f, dist_f], dim=1))
        else:
            out = self.conv_add(cnn_f+dist_f)
        return out

## Nonlocal式
################### fusion11
class Attention(nn.Module):
    def __init__(self, in_c, zin_c):
        super().__init__()
        self.in_c = in_c
        self.zin_c = zin_c

        self.Q = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.K = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=in_c, kernel_size=1),
            nn.BatchNorm1d(in_c)
        )
        # self.V = nn.Conv1d(in_channels=zin_c, out_channels=zin_c, kernel_size=1)
        self.W = nn.Conv1d(in_channels=zin_c, out_channels=zin_c, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, z):
        '''x: CNN feature: b,1,d
           y: Dist feature: b,c,d
           z: CNN or Dist feature depand on operation
        '''
        ## expand dimention
        B, C, dim = y.size()

        ## calc weight(or similarity)
        query = self.Q(x).permute(0, 2, 1).expand(B, dim, C)  # B, D, C
        key = self.K(y)  # B, C, D
        energy = torch.matmul(query, key)  # B, D, D
        attention = self.softmax(energy)

        # use V or not
        # value = self.V(z)
        value = z  # B, 1/C, D

        # permute是为了让value与列相乘，列和为1, here, out can be the OUTPUT.
        out = torch.matmul(value, attention.permute(0, 2, 1))  # B, 1/C, D

        # op use W or not
        out = self.W(out)

        # residual connection use or not
        out = out + value
        return out

class FeatFusion(nn.Module):
    def __init__(self, in_c, zin_c):
        super().__init__()
        self.F1 = Attention(in_c, zin_c)
        self.F2 = Attention(in_c, in_c)
        self.F3 = Attention(in_c, zin_c)

    def forward(self, x, y):
        ''' output shape is same as x
            x: B, 1, D
            y: B, C, D
            output: B, 1, D
        '''
        # x, y are used to calculate weight, z(aka x) is weighted output
        _x = self.F1(x, y, x)
        _y = self.F2(_x, y, y)
        out = self.F3(_x, _y, _x)
        return out

## AFF
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.conv = nn.Conv1d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        xo = self.conv(xo)
        return xo



"""
Opportunity : 去掉self.K中的BN， MSCResNet中最后一层池化
WISDM：与UCR一致
HAR：与UCR一致
UCR：
"""
class FusionNet(BaseModel):
    def __init__(self, seq_len, n_class, 
            in_channel=1, out_channel=64, 
            kernel_list=[[],[],[]],
            num_experts=[],
            padding = [],
            dilation = [1]*3,
            activate ='p_relu',
            N=1, in_dim=128, out_dim=128):
        super().__init__()
        
        self.cnn_feature_extractor = MSCResNet(seq_len, n_class, in_channel, out_channel,
                                                kernel_list, num_experts, padding, dilation, activate)
        self.dist_feature_extractor = EmbedNet(n_class, N, in_dim, out_dim)

        # conv fusion
        # self.fusion_module = ConvFuse(n_class, mode='cat')
        # my fusion
        self.fusion_module = FeatFusion(n_class, 1)
        # aff
        # self.fusion_module = AFF(n_class) 

        self.fc = nn.Linear(out_dim, n_class)
 
    def fusion_feature(self, cnn_feat, dist_feat):
        """ cnn_feat: B, D
            dis_feat: B, C, D
        """
        cnn_feat = cnn_feat.unsqueeze(1)
        fusion_feat = self.fusion_module(cnn_feat, dist_feat)
        return fusion_feat.squeeze(1)


    def forward(self, x, label=None):
        cnn_y, cnn_feat = self.cnn_feature_extractor(x)
        similarity, dist_feat, (embed, proxies) = self.dist_feature_extractor(cnn_feat, label)
        fusion_feat = self.fusion_feature(cnn_feat, dist_feat)
        y = self.fc(fusion_feat)
        # dump_embedding(proxies, embed, label, y)
        return (y, cnn_y, similarity), (fusion_feat, cnn_feat, dist_feat), (embed, proxies)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def elementwise_l2_dist(x, y):
    # x: embed
    # y: proxies
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x-y, 2)

def dump_embedding(proto_embed, sample_embed, labels=None, logits=None, dump_file='./plot/embeddings.txt'):
    import numpy as np
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    if labels is not None:
        labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                                labels.squeeze().cpu().detach().numpy()), axis=0)
        logits = torch.argmax(logits, 1)
        logits = np.concatenate((np.asarray([i for i in range(nclass)]),
                                logits.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            if labels is not None:
                label = str(labels[i])
                logit = str(logits[i])
                line = label + "," + logit + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            else:
                line = ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')