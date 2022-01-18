import torch
from torch.nn.functional import pad
import models
from models.modules import MultiScaleCondConv1d
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS, FlopCountAnalysis, flop_count

channel = 1
seq_len = 150

fusion = models.FusionNet(seq_len, 2, channel, 64,
                         kernel_list=[
                             [3,5,7],
                             [3,5,7],
                             [3,5,7]
                         ],
                         num_experts=[3]*3,
                         padding=[3,3,3])

od = models.OrdinaryFusionNet(seq_len, 2, channel, 64,
                         kernel_list=[7,7,7],
                         padding=[3,3,3])

model = fusion

# # 分析FLOPs
flops = FlopCountAnalysis(model, torch.randn(1, channel, seq_len))
print("FLOPs: ", flops.total())

# 分析parameters
print(parameter_count_table(model))