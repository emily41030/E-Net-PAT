# -*- coding: UTF-8 -*-
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
import utils
from torchvision.transforms import *
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
import torchvision
import test_v2
from loss import *
from data_zssr import DataSampler
import argparse


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(img, sr_factor, num_batches, learning_rate, crop_size, img_path, img_name, gpu_mode):
    model = Net(3, 64, 3, 1)
    model.load_state_dict(torch.load('epoch0_70.pkl'))
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = DataSampler(img, sr_factor, crop_size)
    if gpu_mode:
        model.cuda()
        loss.cuda()

    for iter, (hr, lr) in enumerate(sampler.generate_data()):
        lr = Variable(lr).cuda()
        hr = Variable(hr).cuda()

        # output = model(lr) + lr
        output = model(lr)

        error = loss(output, hr)

        cpu_loss = error.data.cpu().numpy()[0]

        if iter > 0 and iter % 10000 == 0:
            learning_rate = learning_rate / 10
            adjust_learning_rate(optimizer, new_lr=learning_rate)
            print("Learning rate reduced to {lr}".format(lr=learning_rate))

        error.backward()
        optimizer.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=10000,
                        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=128,
                        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Base learning rate for Adam')
    parser.add_argument('--factor', type=int, default=3,
                        help='Interpolation factor.')
    parser.add_argument('--img', type=str, help='Path to input img')
    parser.add_argument('--gpu_mode', type=bool, default="True")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    path = 'Result_train2014_9000/EnhanceNet_PA epoch50~200 batch16 lr0.0001 overlap0 patch16 loss_F=MSE period3/test_result/License_plate'
    path_out = '/zssr_out'
    path_save = "/zssr_save/"

    # if not os.path.isdir(path_save):
    #     os.mkdir(path_save)

    #NameList = os.listdir(path_save)
    # for files in os.walk(path_save):
    #     namearr[]

    for subdir, dirs, files in os.walk(path):
        for f in files:
            print(f)
            newName = f.split(".")
            # print(cur_path)
            # os.rename(path+'/'+f, path+'/'+newName[0]+'_resize.jpg')

            '''若圖片已處理過就掠過
            if (f is NameList[x] for x in NameList):
                print("Image Exist")
            else:
            '''

            args = get_args()
            # img = PIL.Image.open(args.img)
            img = PIL.Image.open(path+'/'+f)

            train(img, args.factor, args.num_batches, args.lr,
                  args.crop, path_save+newName[0], newName[0], args.gpu_mode)

            if not os.path.isdir(path_out):
                os.mkdir(path_out)
            #test(model, img, args.factor, path_out+newName[0])
