import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import torch.nn.functional as F
from Loss import Correlation_Loss,SSIM,ReconLoss,PredLoss
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision
import numpy as np
import torch.nn.functional as F
from PIL import Image
from model import SUP,Downsampler
import umap
from sklearn.preprocessing import MinMaxScaler
from utils import *

def SUP_for_emb(imgg,imgg2,imgg3,n_factor,save_result,output_file):

    model_G = SUP(input_channel = 3).cuda()

    optimizer_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    PreLoss = PredLoss(n_factor,3)
    RecLoss = ReconLoss(n_factor,3)
    CorLoss = Correlation_Loss()

    for j in range(5000):

        model_G.zero_grad()

        output, y_stn, y = model_G(imgg, imgg2)

        if j < 1000:

            pred_loss = PreLoss(y, imgg3)
            pred_loss.backward(retain_graph=True)
            print('warm up :' + str(j) + '  ' + str(pred_loss.item()))

        else:

            rec_loss = RecLoss(output, imgg3)
            corr_loss = CorLoss(output, y_stn)
            pred_loss = PreLoss(y, imgg3)

            loss_total = rec_loss + corr_loss + pred_loss
            loss_total.backward(retain_graph=True)

        optimizer_G.step()

        if j % 100 == 0 and j > 1000:

            print('Loss value:' + str(j - 1000) + '  ' + str(loss_total.item()))

            if save_result ==True:

                save_pic(output,j,output_file)

def SUP_for_ion(imgg, imgg2, imgg3, imgg4, imgg5, n_factor,save_result,output_file):

    PreLoss = PredLoss(n_factor,3)
    model_G = SUP(input_channel=3).cuda()
    optimizer_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for j in range(1000):

        model_G.zero_grad()

        _, yy_stn, yyy = model_G(imgg4, imgg2)

        pred_loss = PreLoss(yyy, imgg5)

        pred_loss.backward(retain_graph=True)
        optimizer_G.step()

        print('warm up :' + str(j) + '  ' + str(pred_loss.item()))

    model_G = SUP(input_channel=1).cuda()
    optimizer_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    PreLoss = PredLoss(n_factor,1)
    RecLoss = ReconLoss(n_factor,1)

    CorLoss = Correlation_Loss()
    yy_stn = yy_stn.detach()
    for j in range(500):

        model_G.zero_grad()

        output, y_stn, y = model_G(imgg, yy_stn)

        rec_loss = PreLoss(y, imgg3)
        corr_loss = CorLoss(output, y)
        rec_loss2 = RecLoss(output, imgg3)
        loss_total = rec_loss + corr_loss + rec_loss2

        loss_total.backward(retain_graph=True)
        optimizer_G.step()

        if j % 20 == 0 and j > 20:
            print('Loss value:' + str(j) + '  ' + str(loss_total.item()))

            if save_result ==True:
                save_txt(output, j, output_file)

