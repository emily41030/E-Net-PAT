# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import copy
import torchvision.models as models
from base_networks import *
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from torchvision import transforms


def crop_to_patch(self, image):
    patch_size = self.patch_size
    interval = patch_size-self.overlap

    crop_img = None
    for w in range(0, self.crop_size, interval):
        for h in range(0, self.crop_size, interval):
            # [batch,3,16,16]
            patch = image[:, :, w:w+patch_size, h:h+patch_size]
            if w == 0 and h == 0:
                crop_img = patch
            else:
                crop_img = torch.cat(
                    ([crop_img, patch]), 0)
    # if crop_img :
    #     print("Error crop image to patch")
    return crop_img


def mse_loss(input, target):
    return (torch.sum((input - target)**2) / input.data.nelement())


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


class Loss:
    def loss_op(self, self_E,  recon_image, x_):
        loss_a = 0
        loss_output_m2 = 0
        loss_output_m5 = 0
        style_loss = 0
        loss_G = 0
        loss_T = []
        if 'A' in self_E.model_loss:
            if self_E.loss_F == "BCEWithLogitsLoss":
                loss_a = self_E.criterion_GAN(
                    self_E.discriminator(recon_image), self_E.discriminator(x_))
                loss_G = self_E.criterion_GAN(
                    self_E.discriminator(recon_image), self_E.fake)
            elif self_E.loss_F == "MSE":
                loss_a = self.mse_loss(
                    self_E.discriminator(recon_image), self_E.discriminator(x_))
                loss_G = self.mse_loss(
                    self_E.discriminator(recon_image), self_E.fake)

        if 'P' in self_E.model_loss:
            # print("creat P loss")
            ############## VGG maxpooling_2 #################
            recon_loss_m2 = self_E.VGG_m2_model(recon_image)
            xs_loss_m2 = self_E.VGG_m2_model(x_)

            for re_m2, xs_m2 in zip(recon_loss_m2, xs_loss_m2):
                loss_output_m2 += mse_loss(re_m2, xs_m2)
            ############## VGG maxpooling_5 #################
            recon_loss_m5 = self_E.VGG_m5_model(recon_image)
            xs_loss_m5 = self_E.VGG_m5_model(x_)

            for re_m5, xs_m5 in zip(recon_loss_m5, xs_loss_m5):
                loss_output_m5 += mse_loss(re_m5, xs_m5)

        if 'T' in self_E.model_loss:
            #################################################
            # 輸入整張圖到model，將conv1_1，conv2_1，conv3_1的feature map取出
            # 並切成16*16大小做loss運算
            ###############################################
            vgg = Vgg19(requires_grad=False).cuda(
            ) if self_E.gpu_mode else Vgg19(requires_grad=False)
            style_transform = transforms.Compose(
                [(transforms.Lambda(lambda x: x.mul(255)))])
            style = style_transform(x_)
            style = style.repeat(self_E.batch_size, 1, 1, 1).cuda(
            ) if self_E.gpu_mode else style.repeat(self_E.batch_size, 1, 1, 1)

            features_style = vgg(normalize_batch(style))
            gram_style = [gram_matrix(y) for y in features_style]
            
            n_batch = len(x_)
            y = normalize_batch(recon_image)
            features_y = vgg(y.detach())
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])

        return loss_a, loss_output_m2, loss_output_m5,style_loss, loss_G
