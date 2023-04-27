import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision
from model import Downsampler

class PredLoss(torch.nn.Module):
    def __init__(self,n_factor,n_planes):
        super(PredLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.downsample = Downsampler(factor=n_factor, kernel_type='lanczos2', phase=0.5,preserve_size=True,n_planes=n_planes).type(torch.cuda.FloatTensor)

    def forward(self, x, y):
        x = self.downsample(x)
        return self.mse(x,y)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def info_nce_loss(features):

    labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / 0.07
    return logits, labels

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = torchvision.models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break

        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)

        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

class Correlation_Loss(torch.nn.Module):

    def patch_correlation(self, patch,index_zero):

        _,k,m,n = patch.shape

        patch_fallten = patch.reshape(k, m*n)

        patch_expand = patch_fallten.expand(m*n,k,m*n)
        patch_expand_2 = patch_expand.permute(2,1,0)
        corr = torch.pow(patch_expand - patch_expand_2,2)
        return_cor2 = torch.sum(corr,dim = 1)

        return_cor2[:,index_zero] = 0
        return_cor2[index_zero,:] = 0

        return return_cor2

    def __init__(self):
        super(Correlation_Loss,self).__init__()
        self.mse = nn.L1Loss(reduction='elementwise_mean')

    def forward(self, tensor_msi, tensor_he, window_size = 16,batch = 800):

        _,u,m,n = tensor_msi.shape

        return_loss = 0

        for kk in range(batch):

            i = np.random.randint(low=0, high=m - window_size)
            j = np.random.randint(low=0, high=n - window_size)
            patch_msi = tensor_msi[:, :u//2, i:i + window_size, j: j + window_size]
            patch_he = tensor_he[:, :, i:i + window_size, j:j + window_size]

            _, k, mm, nn = patch_msi.shape

            patch_msi_fallten = patch_msi.reshape(k, mm * nn)
            patch_sum = torch.sum(patch_msi_fallten, 0)
            index_zero = (patch_sum < 0.05).nonzero()

            _, k, mm, nn = patch_he.shape

            patch_he_fallten = patch_he.reshape(k, mm * nn)
            patch_sum = torch.sum(patch_he_fallten, 0)
            index_zero2 = (patch_sum < 0.05).nonzero()

            index_zero = index_zero2

            patch_cor_msi = self.patch_correlation(patch_msi,index_zero)
            patch_cor_he = self.patch_correlation(patch_he,index_zero)
            patch_cor_he = patch_cor_he.detach()

            patch_loss = self.mse(patch_cor_msi, patch_cor_he)
            return_loss = return_loss + patch_loss

        return return_loss / (batch//5)

# A = torch.rand([1,3,56,70]).cuda()
# B = torch.rand([1,3,56,70]).cuda()
# print(A)
# print(Correlation_Loss().cuda()(A,B))

# A = torch.rand([1,3,56,70]).cuda()
# B = torch.rand([1,3,56,70]).cuda()
# print(A)
# print(Correlation_Loss().cuda()(A,B))

class ReconLoss(torch.nn.Module):
    def __init__(self,n_factor,n_planes):
        super(ReconLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.downsample = Downsampler(factor=n_factor, kernel_type='lanczos2', phase=0.5,preserve_size=True,n_planes=n_planes * 2).type(torch.cuda.FloatTensor)

    def forward(self, x, y):
        _,u,_,_ = x.shape
        x = self.downsample(x)
        return self.mse(x[:,:u//2,:,:],y) + self.mse(x[:,u//2:,:,:],y)



