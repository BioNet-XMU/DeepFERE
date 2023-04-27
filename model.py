import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

class Unet(nn.Module):
    def __init__(self,num_input_channels=2, num_output_channels=3,
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need1x1_up=True):

        super(Unet,self).__init__()

        upsample_mode = 'bilinear'
        skip_n33d = 128
        num_scales = 5
        skip_n11 = 5
        skip_n33u = 128
        pad = 'reflection'
        act_fun = 'LeakyReLU'
        downsample_mode = 'stride'

        num_channels_down = [skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d
        num_channels_up = [skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u
        num_channels_skip = [skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11
        upsample_mode = upsample_mode
        downsample_mode = downsample_mode
        need_sigmoid = True
        need_bias = True
        pad = pad
        act_fun = act_fun
        self.ch = num_input_channels

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down)

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales

        if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
            downsample_mode = [downsample_mode] * n_scales

        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
            filter_size_down = [filter_size_down] * n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales

        last_scale = n_scales - 1

        cur_depth = None

        self.model = nn.Sequential()
        self.model_tmp = self.model

        input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                self.model_tmp.add(Concat(1, skip, deeper))
            else:
                self.model_tmp.add(deeper)

            if num_channels_skip[i] != 0:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(act(act_fun))

            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                            downsample_mode=downsample_mode[i]))
            deeper.add(act(act_fun))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(act(act_fun))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]

            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

            self.model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            self.model_tmp.add(act(act_fun))

            if need1x1_up:
                self.model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                self.model_tmp.add(act(act_fun))

            input_depth = num_channels_down[i]
            self.model_tmp = deeper_main

        self.model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            self.model.add(nn.Sigmoid())

        self.revise = nn.Sequential(
            nn.Conv2d(self.ch, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.ch, kernel_size=3, stride=1, padding=1)
        ).cuda()


    def forward(self,x):

        x = self.model(x[:,:self.ch,:,:])
        output = self.revise(x)

        return torch.cat([x,output],dim = 1)


class Downsampler(nn.Module):

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None,
                 preserve_size=False):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'

        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:

            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)

            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x

        x = self.downsampler_(x)

        return x


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val


    else:
        assert False, 'wrong method name'

    kernel /= kernel.sum()

    return kernel


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class SUP(nn.Module):
    def __init__(self, input_channel):
        super(SUP, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),

            nn.Conv2d(10, 8, kernel_size=1),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),

        )

        self.fc_loc = nn.Sequential(
            nn.Linear(1 * 8, 16),
            nn.ReLU(True),
            nn.Linear(16, 3 * 2)
        ).cuda()

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.head = ConvBlock(3, 64, 3, 1, 1)

        self.body = nn.Sequential()
        for i in range(3):
            block = ConvBlock(64, 64, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = ConvBlock(64, input_channel, 3, 1, 1)

        self.Unet = Unet(input_channel, input_channel).cuda()

    def stn(self, x):
        xs = self.localization(x)
        xs = torch.nn.AdaptiveAvgPool2d((1, 1))(xs)

        xs = xs.contiguous().view(1, 1 * 8)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, y):

        y_stn = self.stn(y)

        y = self.head(y_stn)
        y = self.body(y)
        y = self.tail(y)

        x = torch.cat([x, y], dim=1)

        output = self.Unet(x)

        return output, y_stn, y
