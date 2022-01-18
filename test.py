import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

from dataloaders import *
from utils import augmentation as aug
from models import *
import pandas as pd

# data_lst = [
#     'Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','Car','CBF','ChlorineConcentration','CinCECGTorso','Coffee',
#     'Computers','CricketX','CricketY','CricketZ','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup',
#     'DistalPhalanxOutlineCorrect','DistalPhalanxTW','Earthquakes','ECG200','ECG5000','ECGFiveDays','ElectricDevices',
#     'FaceAll','FaceFour','FacesUCR','FiftyWords','Fish','FordA','FordB','GunPoint','Ham','HandOutlines','Haptics',
#     'Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lightning2','Lightning7',
#     'Mallat','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW',
#     'MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','OliveOil','OSULeaf','PhalangesOutlinesCorrect',
#     'Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices',
#     'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface1','SonyAIBORobotSurface2',
#     'StarLightCurves','Strawberry','SwedishLeaf','Symbols','SyntheticControl','ToeSegmentation1','ToeSegmentation2',
#     'Trace','TwoLeadECG','TwoPatterns','UWaveGestureLibraryAll','UWaveGestureLibraryX','UWaveGestureLibraryY',
#     'UWaveGestureLibraryZ','Wafer','Wine','WordSynonyms','Worms','WormsTwoClass','Yoga'
# ]

# UCRArchive_2018
data_dir = 'Data/UCRArchive_2018'
subset = 'Adiac'
gpu = 0

model_path = 'saved/ckp-Adiac.pth'
batch_size = 64

val_dataloader, dataset = UCRArchive2018Loader(
        data_dir, subset, batch_size, [aug.expand_dim, aug.to_torch], False, False)

model = FusionNet(dataset.seq_len, dataset.n_cls, 1, 64,
                    kernel_list=[[3,5,7],[3,5,7],[3,5,7]],
                    num_experts=[3]*3,
                    padding=[3]*3)

state = torch.load(model_path, map_location='cpu')
model.load_state_dict(state['state_dict'])

model = model.cuda(gpu)
model.eval()

correct = 0
total = 0

s = time.time()
with torch.no_grad():
    for data in val_dataloader:
        data, labels = data
        data, labels = data.cuda(gpu), labels.cuda(gpu)

        # for fusionnet
        (y, _, _), (_, _, _), (_, _) = model(data, labels)

        # 分类
        pred = torch.argmax(y, dim=1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()


print(time.time()-s, total, '|', correct)
print('Accuracy: %.4f' % (correct / total))