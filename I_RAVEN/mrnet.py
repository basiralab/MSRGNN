# Adapted from MRNet code at https://github.com/yanivbenny/MRNet

from data_utility.data_utility_norm import dataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(x + self.bn2(self.conv2(out)))

        return out
    

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class MRNet(nn.Module):
    def __init__(
            self,
            use_meta=False,
            row_col=True,
            dropout=False,
            force_bias=False,
            relu_before_reduce=False,
            reduce_func='sum',
            levels='111',
            multihead=False,
            big=False,
    ):
        super(MRNet, self).__init__()
        self.use_meta = use_meta
        self.relu_before_reduce = relu_before_reduce
        self.levels = levels
        print(f'LEVELS: {self.levels}')

        if dropout:
            _dropout = {
                'high': 0.1,
                'mid': 0.1,
                'low': 0.1,
                'mlp': 0.5,
            }
        else:
            _dropout = {
                'high': 0.,
                'mid': 0.,
                'low': 0.,
                'mlp': 0.,
            }
        # Perception
        if big:
            self.high_dim, self.high_dim0 = 128, 64
            self.mid_dim, self.mid_dim0 = 256, 128
            self.low_dim, self.low_dim0 = 512, 256
        else:
            self.high_dim, self.high_dim0 = 64, 32
            self.mid_dim, self.mid_dim0 = 128, 64
            self.low_dim, self.low_dim0 = 256, 128

        self.perception_net_high = nn.Sequential(
            nn.Conv2d(1, self.high_dim0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.high_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['high']),
            nn.Conv2d(self.high_dim0, self.high_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.high_dim),
            nn.ReLU(inplace=True))

        self.perception_net_mid = nn.Sequential(
            nn.Conv2d(self.high_dim, self.mid_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['mid']),
            nn.Conv2d(self.mid_dim0, self.mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
            )

        self.perception_net_low = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.low_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['low']),
            nn.Conv2d(self.low_dim0, self.low_dim, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.low_dim),
            nn.ReLU(inplace=True)
            )

        self.g_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),
                                             conv3x3(3 * self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim))
        self.g_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 5, 5)),
                                            conv3x3(3 * self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim))
        self.g_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 1, 1)),
                                            conv1x1(3 * self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim))

        self.reduce_func = reduce_func

        self.conv_row_high = conv3x3(self.high_dim, self.high_dim)
        self.bn_row_high = nn.BatchNorm2d(self.high_dim)
        self.conv_col_high = conv3x3(self.high_dim, self.high_dim) if row_col else self.conv_row_high
        self.bn_col_high = nn.BatchNorm2d(self.high_dim, ) if row_col else self.bn_row_high

        self.conv_row_mid = conv3x3(self.mid_dim, self.mid_dim)
        self.bn_row_mid = nn.BatchNorm2d(self.mid_dim)
        self.conv_col_mid = conv3x3(self.mid_dim, self.mid_dim) if row_col else self.conv_row_mid
        self.bn_col_mid = nn.BatchNorm2d(self.mid_dim) if row_col else self.bn_row_mid

        self.conv_row_low = conv1x1(self.low_dim, self.low_dim)
        self.bn_row_low = nn.BatchNorm2d(self.low_dim)
        self.conv_col_low = conv1x1(self.low_dim, self.low_dim) if row_col else self.conv_row_low
        self.bn_col_low = nn.BatchNorm2d(self.low_dim) if row_col else self.bn_row_low

        if not force_bias and not relu_before_reduce:
            if reduce_func not in ['sum']:
                self.bn_row_high.register_parameter('bias', None)
                self.bn_col_high.register_parameter('bias', None)
                self.bn_row_mid.register_parameter('bias', None)
                self.bn_col_mid.register_parameter('bias', None)
                self.bn_row_low.register_parameter('bias', None)
                self.bn_col_low.register_parameter('bias', None)
            if reduce_func in ['prodi', 'prodi3']:
                self.bn_row_high.register_parameter('weight', None)
                self.bn_col_high.register_parameter('weight', None)
                self.bn_row_mid.register_parameter('weight', None)
                self.bn_col_mid.register_parameter('weight', None)
                self.bn_row_low.register_parameter('weight', None)
                self.bn_col_low.register_parameter('weight', None)

        self.mlp_dim_high = self.mlp_dim_mid = self.mlp_dim_low = 0
        if self.levels[0] == '1':
            if big:
                self.mlp_dim_high = 256
            else:
                self.mlp_dim_high = 128
            self.res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                      downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                               nn.BatchNorm2d(2 * self.high_dim)
                                                               )
                                      )

            self.res2_high = ResBlock(2 * self.high_dim, self.mlp_dim_high, stride=2,
                                      downsample=nn.Sequential(conv1x1(2 * self.high_dim, self.mlp_dim_high, stride=2),
                                                               nn.BatchNorm2d(self.mlp_dim_high)
                                                               )
                                      )

        if self.levels[1] == '1':
            if big:
                self.mlp_dim_mid = 256
            else:
                self.mlp_dim_mid = 128
            self.res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                     downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                              nn.BatchNorm2d(2 * self.mid_dim)
                                                              )
                                     )

            self.res2_mid = ResBlock(2 * self.mid_dim, self.mlp_dim_mid, stride=2,
                                     downsample=nn.Sequential(conv1x1(2 * self.mid_dim, self.mlp_dim_mid, stride=2),
                                                              nn.BatchNorm2d(self.mlp_dim_mid)
                                                              )
                                     )

        if self.levels[2] == '1':
            if big:
                self.mlp_dim_low = 256
            else:
                self.mlp_dim_low = 128
            self.res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                          nn.BatchNorm2d(self.mlp_dim_low),
                                          nn.ReLU(inplace=True))
            self.res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_dim = self.mlp_dim_high + self.mlp_dim_mid + self.mlp_dim_low
        self.mlp = nn.Sequential(nn.Linear(self.mlp_dim, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(_dropout['mlp']),
                                 nn.Linear(256, 128, bias=False),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 1, bias=True))

        if use_meta:
            self.mlp_meta = nn.Sequential(nn.Linear(self.mlp_dim, 256, bias=False),
                                          nn.BatchNorm1d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(_dropout['mlp']),
                                          nn.Linear(256, 128, bias=False),
                                          nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, use_meta, bias=True))

        self.multihead = multihead
        if multihead:
            self.mlp_high = nn.Sequential(nn.Linear(self.mlp_dim_high, 256, bias=False),
                                          nn.BatchNorm1d(256),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(_dropout['mlp']),
                                          nn.Linear(256, 128, bias=False),
                                          nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, 1, bias=True))
            self.mlp_mid = nn.Sequential(nn.Linear(self.mlp_dim_mid, 256, bias=False),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(_dropout['mlp']),
                                         nn.Linear(256, 128, bias=False),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(128, 1, bias=True))
            self.mlp_low = nn.Sequential(nn.Linear(self.mlp_dim_low, 256, bias=False),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(_dropout['mlp']),
                                         nn.Linear(256, 128, bias=False),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(128, 1, bias=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def triples(self, input_features):
        N, K, C, H, W = input_features.shape
        K0 = K - 8
        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row1_features = input_features[:, 0:3, :, :, :]  # N, 3, 64, 20, 20
        row2_features = input_features[:, 3:6, :, :, :]  # N, 3, 64, 20, 20
        # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H, W)
        # N, 8, 2, 64, 20, 20 - > N, 8, 3, 64, 20, 20
        row3_features = torch.cat((row3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        col1_features = input_features[:, 0:8:3, :, :, :]  # N, 3, 64, 20, 20
        col2_features = input_features[:, 1:8:3, :, :, :]  # N, 3, 64, 20, 20
        # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H, W)
        # N, 8, 2, 64, 20, 20 - > N, 8, 3, 64, 20, 20
        col3_features = torch.cat((col3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        return row1_features, row2_features, row3_features, col1_features, col2_features, col3_features

    def apply_reduce(self, x1, x2, x3):
        if self.relu_before_reduce:
            x1, x2, x3 = F.relu(x1), F.relu(x2), F.relu(x3)

        if self.reduce_func == 'sum':
            x = F.relu(x1 + x2 + x3)
        elif self.reduce_func == 'dist':
            dist12, dist23, dist31 = (x1 - x2).pow(2), (x2 - x3).pow(2), (x3 - x1).pow(2)
            x = 1 - (dist12 + dist23 + dist31)
        elif self.reduce_func == 'dist3':
            x_mean = (x1 + x2 + x3) / 3
            dist12, dist23, dist31 = (x1 - x_mean).pow(2), (x2 - x_mean).pow(2), (x3 - x_mean).pow(2)
            x = 1 - (dist12 + dist23 + dist31)
        elif self.reduce_func == 'dist3-sg':
            x_mean = (x1 + x2 + x3).detach() / 3
            dist12, dist23, dist31 = (x1 - x_mean).pow(2), (x2 - x_mean).pow(2), (x3 - x_mean).pow(2)
            x = 1 - (dist12 + dist23 + dist31)
        elif self.reduce_func == 'prod':
            prod12, prod23, prod31 = x1 * x2, x2 * x3, x3 * x1
            x = prod12 + prod23 + prod31
        elif self.reduce_func == 'prodi':
            prod12, prod23, prod31 = 1 / (x1 * x2 + 0.1), 1 / (x2 * x3 + 0.1), 1 / (x3 * x1 + 0.1)
            x = prod12 + prod23 + prod31
        elif self.reduce_func == 'prodi3':
            x_mean = (x1 + x2 + x3).detach() / 3
            prod12, prod23, prod31 = 1 / (x1 * x_mean + 0.1), 1 / (x2 * x_mean + 0.1), 1 / (x3 * x_mean + 0.1)
            x = prod12 + prod23 + prod31
        return x

    def reduce(self, row_features, col_features, N, K0):
        _, C, H, W = row_features.shape

        row1 = row_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row2 = row_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row3 = row_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_row_features = self.apply_reduce(row1, row2, row3)

        col1 = col_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col2 = col_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col3 = col_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_col_features = self.apply_reduce(col1, col2, col3)

        input_features = final_row_features + final_col_features
        return input_features

    def forward(self, x):
        N, K, C, H, W = x.shape
        K0 = K - 8
        # assert C==1
        x = x.view(-1, K, 80, 80)

        ### Perception Branch
        input_features_high = self.perception_net_high(x.view(-1, 80, 80).unsqueeze(1))
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)

        ### Relation Module
        # High res
        if self.levels[0] == '1':
            row1_cat_high, row2_cat_high, row3_cat_high, col1_cat_high, col2_cat_high, col3_cat_high = \
                self.triples(input_features_high.view(N, K, self.high_dim, 20, 20))

            row_feats_high = self.g_function_high(torch.cat((row1_cat_high, row2_cat_high, row3_cat_high), dim=0))
            row_feats_high = self.bn_row_high(self.conv_row_high(row_feats_high))
            col_feats_high = self.g_function_high(torch.cat((col1_cat_high, col2_cat_high, col3_cat_high), dim=0))
            col_feats_high = self.bn_col_high(self.conv_col_high(col_feats_high))

            reduced_feats_high = self.reduce(row_feats_high, col_feats_high, N, K0)  # N, 8, 64, 20, 20

        # Mid res
        if self.levels[1] == '1':
            row1_cat_mid, row2_cat_mid, row3_cat_mid, col1_cat_mid, col2_cat_mid, col3_cat_mid = \
                self.triples(input_features_mid.view(N, K, self.mid_dim, 5, 5))

            row_feats_mid = self.g_function_mid(torch.cat((row1_cat_mid, row2_cat_mid, row3_cat_mid), dim=0))
            row_feats_mid = self.bn_row_mid(self.conv_row_mid(row_feats_mid))
            col_feats_mid = self.g_function_mid(torch.cat((col1_cat_mid, col2_cat_mid, col3_cat_mid), dim=0))
            col_feats_mid = self.bn_col_mid(self.conv_col_mid(col_feats_mid))

            reduced_feats_mid = self.reduce(row_feats_mid, col_feats_mid, N, K0)  # N, 8, 128, 10, 10

        # Low res
        if self.levels[2] == '1':
            row1_cat_low, row2_cat_low, row3_cat_low, col1_cat_low, col2_cat_low, col3_cat_low = \
                self.triples(input_features_low.view(N, K, self.low_dim, 1, 1))

            row_feats_low = self.g_function_low(torch.cat((row1_cat_low, row2_cat_low, row3_cat_low), dim=0))
            row_feats_low = self.bn_row_low(self.conv_row_low(row_feats_low))
            col_feats_low = self.g_function_low(torch.cat((col1_cat_low, col2_cat_low, col3_cat_low), dim=0))
            col_feats_low = self.bn_col_low(self.conv_col_low(col_feats_low))

            reduced_feats_low = self.reduce(row_feats_low, col_feats_low, N, K0)  # N, 8, 256, 5, 5

        ### Combine
        self.final_high = self.final_mid = self.final_low = None
        final = []
        # High
        if self.levels[0] == '1':
            res1_in_high = reduced_feats_high
            # if self.do_contrast:
            #     res1_in_high = res1_in_high - res1_in_high.mean(dim=1).unsqueeze(1)
            res1_out_high = self.res1_high(res1_in_high.view(N * K0, self.high_dim, 20, 20))
            res2_in_high = res1_out_high.view(N, K0, 2 * self.high_dim, 10, 10)
            # if self.do_contrast:
            #     res2_in_high = res2_in_high - res2_in_high.mean(dim=1).unsqueeze(1)
            out_high = self.res2_high(res2_in_high.view(N * K0, 2 * self.high_dim, 10, 10))
            final_high = self.avgpool(out_high)
            final_high = final_high.view(-1, self.mlp_dim_high)
            final.append(final_high)
            self.final_high = final_high

        # Mid
        if self.levels[1] == '1':
            res1_in_mid = reduced_feats_mid
            # if self.do_contrast:
            #     res1_in_mid = res1_in_mid - res1_in_mid.mean(dim=1).unsqueeze(1)
            res1_out_mid = self.res1_mid(res1_in_mid.view(N * K0, self.mid_dim, 5, 5))
            res2_in_mid = res1_out_mid.view(N, K0, 2 * self.mid_dim, 3, 3)
            # if self.do_contrast:
            #     res2_in_mid = res2_in_mid - res2_in_mid.mean(dim=1).unsqueeze(1)
            out_mid = self.res2_mid(res2_in_mid.view(N * K0, 2 * self.mid_dim, 3, 3))
            final_mid = self.avgpool(out_mid)
            final_mid = final_mid.view(-1, self.mlp_dim_mid)
            final.append(final_mid)
            self.final_mid = final_mid

        # Low
        if self.levels[2] == '1':
            res1_in_low = reduced_feats_low
            # if self.do_contrast:
            #     res1_in_low = res1_in_low - res1_in_low.mean(dim=1).unsqueeze(1)
            res1_out_low = self.res1_low(res1_in_low.view(N * K0, self.low_dim, 1, 1))
            res2_in_low = res1_out_low.view(N, K0, self.mlp_dim_low, 1, 1)
            # if self.do_contrast:
            #     res2_in_low = res2_in_low - res2_in_low.mean(dim=1).unsqueeze(1)
            out_low = self.res2_low(res2_in_low.view(N * K0, self.mlp_dim_low, 1, 1))
            final_low = self.avgpool(out_low)
            final_low = final_low.view(-1, self.mlp_dim_low)
            final.append(final_low)
            self.final_low = final_low

        final = torch.cat(final, dim=1)
        # MLP
        out = self.mlp(final)

        if self.use_meta:
            meta_pred = self.mlp_meta(final)
            out_meta = meta_pred.view(-1, K0, meta_pred.shape[1])
        else:
            out_meta = None

        if self.multihead:
            out_multihead = [self.mlp_high(final_high).view(-1, K0),
                             self.mlp_mid(final_mid).view(-1, K0),
                             self.mlp_low(final_low).view(-1, K0)]
        else:
            out_multihead = None

        return out.view(-1, K0), out_meta, out_multihead
    

def renormalize(images):
    return (images - 0.5) * 2


def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0; total_correct=0; total_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = renormalize(imgs)  # Normalize images to [-1, 1]

        optimizer.zero_grad()
        model_outputs = model(x=imgs)

        if len(model_outputs) == 3:  # If using meta and multihead
            model_output, _, model_output_multihead = model_outputs
        else:
            model_output, _ = model_outputs
            model_output_multihead = None

        loss = criterion(model_output, labels)

        loss_head = [criterion(output, labels, reduction='none') for output in model_output_multihead]
        target_one_hot = torch.zeros_like(model_output_multihead[0])
        target_one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        probs = [target_one_hot * output.detach().sigmoid() + (1 - target_one_hot) * (1 - output.detach().sigmoid()) for output in model_output_multihead]
        e_probs = [prob.exp() for prob in probs]
        e_probs_sum = sum(e_probs)
        weights = [prob / e_probs_sum for prob in e_probs]

        loss_multihead = sum(weights[i] * loss_head[i] for i in range(len(loss_head))).mean()
        loss += loss_multihead

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (model_output.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        
    average_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_samples
    print(f"Epoch Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    # return total_loss / len(loader.dataset), total_correct / total_samples
    return average_loss, accuracy

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); total_loss=0; correct=0; total=0
    for imgs, labels in tqdm(loader, desc="Validating"):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = renormalize(imgs)  # Normalize images to [-1, 1]
        scores = model(x=imgs)[0]
        loss = criterion(scores, labels)
        preds = scores.argmax(dim=1)
        correct += (preds==labels).sum().item(); total += labels.size(0)
        total_loss += loss.item() * imgs.size(0)
    return total_loss/len(loader.dataset), correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def contrast_loss(output, target, reduction='mean', weighted_loss=False):
    labels = torch.zeros_like(output)
    labels.scatter_(1, target.view(-1, 1), 1.0)
    weights = (1 + 6 * labels) / 7 if weighted_loss else None

    return F.binary_cross_entropy_with_logits(output, labels, weight=weights, reduction=reduction)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()

    RAVEN_ROOT_DIR = args.data
    #fix seed for reproducibility
    set_seed(42)
    torch.autograd.set_detect_anomaly(True)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3 # May need tuning
    EPOCHS = 100 
    IMG_SIZE = 80
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    WEIGHT_DECAY = 1e-6
    META_BETA = 0

    PATIENCE = 10


    writer = SummaryWriter(log_dir='runs/RAVEN_MRNET')

    # --- DataLoaders ---
    from types import SimpleNamespace
    args = SimpleNamespace(
        path=RAVEN_ROOT_DIR,
        img_size=IMG_SIZE,
        percent=100,
    )

    transform=None

    train_dataset = dataset(args, mode="train", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'], transform=transform, transform_p=0.0)
    # train_dataset = dataset(args, mode="train", rpm_types=['cs'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    val_dataset = dataset(args, mode="val", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # val_dataset = dataset(args, mode="val", rpm_types=['cs'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    test_dataset = dataset(args, mode="test", rpm_types=['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'])
    # test_dataset = dataset(args, mode="test", rpm_types=['cs'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    model = MRNet(
        use_meta=False,
        # dropout=False,
        # force_bias=False,
        # reduce_func='dist',
        # levels='111',
        multihead=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPSILON, weight_decay=WEIGHT_DECAY) 
    criterion = lambda x, y, reduction='mean': contrast_loss(x, y, 'mean', False)
    scheduler = None

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print(f"Using device: {DEVICE}")
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}:")
        # Reset memory stats if using GPU
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_time = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Train Time: {train_time:.2f} seconds")
        
        mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
        print(f"  Mem Allocated: {mem_allocated:.2f} MB")
        print(f"  Mem Reserved: {mem_reserved:.2f} MB")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        if scheduler:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Mem/allocated", mem_allocated, epoch)
        writer.add_scalar("Mem/reserved", mem_reserved, epoch)
        writer.add_scalar("Time/epoch", train_time, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "./saved_models/best_model_mrnet.pth")
            print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement in validation accuracy for {PATIENCE} epochs. Stopping training.")
                break

    # Save the model state
    torch.save(model.state_dict(), "./saved_models/model_final_mrnet.pth")
    print("Training complete. Model saved as 'model_final_mrnet.pth'.")

    print("Training finished.")

    # --- Testing ---
    print("Testing the model on test dataset...")

    model.load_state_dict(torch.load("./saved_models/best_model_mrnet.pth"))
    model.eval()
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    writer.add_scalar("Loss/test", test_loss, EPOCHS)
    writer.add_scalar("Acc/test", test_acc, EPOCHS)


if __name__ == "__main__":
    main()