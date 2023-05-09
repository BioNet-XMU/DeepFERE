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
from model import SUP_1,SUP_2,Downsampler
import umap
from sklearn.preprocessing import MinMaxScaler
from utils import *

def SUP_for_emb(imgg,imgg2,imgg3,n_factor,save_result,output_file):

    model_1 = SUP_1(input_channel = 3).cuda()
    model_2 = SUP_2(input_channel = 3).cuda()

    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.00002, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.00002, betas=(0.5, 0.999))

    PreLoss = PredLoss(n_factor,3)
    RecLoss = ReconLoss(n_factor,3)
    CorLoss = Correlation_Loss()

    for j in range(6000):

        if j < 1000:

            model_1.zero_grad()
            y, y_stn = model_1(imgg2)
            pred_loss = PreLoss(y, imgg3)
            pred_loss.backward(retain_graph=True)
            optimizer_1.step()
            print('warm up :' + str(j) + '  ' + str(pred_loss.item()))
            torchvision.utils.save_image(y_stn, 'y_stn.jpg')
            torchvision.utils.save_image(y, 'y.jpg')

        else:

            model_1.zero_grad()
            model_2.zero_grad()

            y, y_stn = model_1(imgg2)
            output = model_2(imgg,y)

            rec_loss = RecLoss(output, imgg3)
            pred_loss = PreLoss(y, imgg3)

            loss_total = rec_loss + pred_loss
            loss_total.backward(retain_graph=True)

            optimizer_1.step()
            optimizer_2.step()

            y = y.detach()
            output = model_2(imgg, y)
            corr_loss = CorLoss(output, y_stn)
            corr_loss.backward(retain_graph=True)
            optimizer_2.step()

        if j % 100 == 0 and j > 1000:

            print('Loss value:' + str(j - 1000) + '  ' + str(loss_total.item()))

            if save_result ==True:

                save_pic(output,j,output_file)

def SUP_for_ion(imgg, imgg2, imgg3, imgg4, imgg5, n_factor,save_result,output_file):

    PreLoss = PredLoss(n_factor,3)
    model_1 = SUP_1(input_channel=3).cuda()
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for j in range(1000):

        model_1.zero_grad()

        yy, yy_stn = model_1(imgg2)

        pred_loss = PreLoss(yy, imgg5)

        pred_loss.backward(retain_graph=True)
        optimizer_1.step()

        print('warm up :' + str(j) + '  ' + str(pred_loss.item()))

    model_1 = SUP_1(input_channel=1).cuda()
    model_2 = SUP_2(input_channel=1).cuda()

    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.0002, betas=(0.5, 0.999))

    PreLoss = PredLoss(n_factor,1)
    RecLoss = ReconLoss(n_factor,1)

    CorLoss = Correlation_Loss()

    for j in range(500):

        model_1.zero_grad()
        model_2.zero_grad()

        y, y_stn = model_1(imgg2)
        output = model_2(imgg, y)

        Pred_Loss = PreLoss(y, imgg3)
        rec_loss2 = RecLoss(output, imgg3)

        loss_total = Pred_Loss + rec_loss2

        loss_total.backward(retain_graph=True)

        optimizer_1.step()
        optimizer_2.step()

        y = y.detach()
        output = model_2(imgg, y)
        corr_loss = CorLoss(output, y)
        corr_loss.backward(retain_graph=True)
        optimizer_2.step()

        if j % 20 == 0 and j > 20:
            print('Loss value:' + str(j) + '  ' + str(loss_total.item()))

            if save_result ==True:
                save_txt(output, j, output_file)

