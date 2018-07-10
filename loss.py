# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import copy
import torchvision.models as models
from base_networks import *
import torch.nn.functional as F
import tensorflow as tf
import numpy as np


def mse_loss(input, target):
    return (torch.sum((input - target)**2) / input.data.nelement())


def normalize(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    return v / tf.reduce_mean(v, axis=[1, 2, 3], keep_dims=True)


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    v = tf.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return tf.matmul(v, v, transpose_a=True)


def tf_op(patch_size, x):
    # [256,64,16,16]  ->  [256,16,16,64]
    x = torch.transpose(x, 1, 3)
    a, b, crop_size, c = x.size()
    # x.numpy()  <-  pytorch_Tensor 轉 numpy
    # tf.convert_to_tensor()   <- numpy 轉 tf
    tf_x = tf.convert_to_tensor(x.cpu().numpy())

    # tf_x = normalize(tf_x)
    assert crop_size % patch_size == 0 and crop_size % patch_size == 0

    # [b * ?, h/p, w/p, c] [32,128,128,64]->[8192,8,8,64]
    tf_x = tf.space_to_batch_nd(
        tf_x, [patch_size, patch_size], [[0, 0], [0, 0]])
    # [p, p, b, h/p, w/p, c]  [8192,8,8,64]->[16,16,32,8,8,64]
    tf_x = tf.reshape(tf_x, [patch_size, patch_size, -1,
                             crop_size // patch_size, crop_size // patch_size, c])
    # [b * ?, p, p, c]  [16,16,32,8,8,64]->[32,8,8,16,16,64]
    tf_x = tf.transpose(tf_x, [2, 3, 4, 0, 1, 5])
    patches_tf_x = tf.reshape(tf_x, [-1, patch_size, patch_size, c])

    return patches_tf_x


def T_loss_op(self_E, recon, x):
    patch_size = self_E.patch_size
    recon_conv1_1 = self_E.conv1_1(recon).detach()
    x_conv1_1 = self_E.conv1_1(x)
    recon_conv1_1 = tf_op(patch_size, recon_conv1_1)
    x_conv1_1 = tf_op(patch_size, x_conv1_1)
    loss_conv1_1 = tf.losses.mean_squared_error(
        gram_matrix(recon_conv1_1),
        gram_matrix(x_conv1_1),
        reduction=tf.losses.Reduction.MEAN
    )

    recon_conv2_1 = self_E.conv2_1(recon).detach()
    x_conv2_1 = self_E.conv2_1(x)
    recon_conv2_1 = tf_op(patch_size, recon_conv2_1)
    x_conv2_1 = tf_op(patch_size, x_conv2_1)
    loss_conv2_1 = tf.losses.mean_squared_error(
        gram_matrix(recon_conv2_1),
        gram_matrix(x_conv2_1),
        reduction=tf.losses.Reduction.MEAN
    )

    recon_conv3_1 = self_E.conv3_1(recon).detach()
    x_conv3_1 = self_E.conv3_1(x)
    recon_conv3_1 = tf_op(patch_size, recon_conv3_1)
    x_conv3_1 = tf_op(patch_size, x_conv3_1)
    loss_conv3_1 = tf.losses.mean_squared_error(
        gram_matrix(recon_conv3_1),
        gram_matrix(x_conv3_1),
        reduction=tf.losses.Reduction.MEAN
    )
    # loss = loss_conv1_1 + loss_conv2_1+loss_conv3_1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)):
        tf.global_variables_initializer()
        loss_conv1_1 = loss_conv1_1.eval()
        loss_conv2_1 = loss_conv2_1.eval()
        loss_conv3_1 = loss_conv3_1.eval()
        # sess.run(tf.global_variables_initializer())
        # tf 轉 numpy

    # numpy 轉 pytorch_Tensor
    loss_conv1_1 = torch.from_numpy(np.array(loss_conv1_1, dtype=np.float32))
    loss_conv2_1 = torch.from_numpy(np.array(loss_conv2_1, dtype=np.float32))
    loss_conv3_1 = torch.from_numpy(np.array(loss_conv3_1, dtype=np.float32))
    # loss= Variable(loss)
    tf.reset_default_graph()
    return loss_conv1_1, loss_conv2_1, loss_conv3_1


class Loss:
    def loss_op(self, self_E,  recon_image, x_):
        loss_a = 0
        loss_output_m2 = 0
        loss_output_m5 = 0
        style_score = 0
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
            loss_conv1_1, loss_conv2_1, loss_conv3_1 = T_loss_op(
                self_E, recon_image, x_)

            style_score = loss_conv1_1*0.3+loss_conv2_1+loss_conv3_1
            if self_E.gpu_mode:
                style_score = style_score.cuda()
            # temp = style_score*1e-6
            # if not style_score==0:
            #     print("style=%.4f   style=%.4f" % (style_score, temp))
            # style_score = loss_conv1_1*0.3s
            # tf.reset_default_graph()
            loss_T.append(loss_conv1_1)
            loss_T.append(loss_conv2_1)
            loss_T.append(loss_conv3_1)
        return loss_a, loss_output_m2, loss_output_m5, style_score, loss_G, loss_T
