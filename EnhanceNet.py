# -*- coding: UTF-8 -*-
import os
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import utils
from logger import Logger
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


class EnhanceNet(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.data_dir = args.data_dir
        self.gpu_mode = args.gpu_mode
        self.overlap = args.overlap
        self.pre_epochs = args.previous_epochs
        self.patchloss = args.patchloss
        self.model_loss = args.model_loss
        self.patch_size = args.patch_size
        self.D_period = args.D_period
        self.pretrain_E_model = args.pretrain_E_model
        self.pretrain_D_model = args.pretrain_D_model
        self.loss_F = args.loss_F
        self.save_dir = os.path.join(args.save_dir)
        self.pretrain_path = ""
        utils.save_dir_name(self)

    def load_dataset(self, dataset, is_train=True):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if is_train:
            print('Loading train datasets...')
            train_set = get_training_set(
                self.data_dir, dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        else:
            print('Loading test datasets...')
            test_set = get_test_set(
                self.data_dir, dataset, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def mse_loss(self, input, target):
        return (torch.sum((input - target)**2) / input.data.nelement())

    def train(self):
        # networks
        self.model = Net(3, 64, 3, 1)
        LossNet.creat_loss_Net(self)
        if self.gpu_mode:
            self.model.cuda()

        if self.pre_epochs == "0":
            # weigh initialization
            self.model.weight_init()
        else:
            # (self, is training or not, is E_model(True) or D_model(False))
            #utils.load_model(self, True, True)
            self.model.load_state_dict(torch.load(self.pretrain_E_model))

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)

        self.valid = None
        self.fake = None
        if 'A' in self.model_loss:
            self.discriminator = Discriminator().cuda() if self.gpu_mode else Discriminator()
            Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.Tensor
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            if self.pre_epochs != "0":
                #utils.load_model(self, True, False)
                self.discriminator.load_state_dict(
                    torch.load(self.pretrain_D_model))
            if self.loss_F == "BCEWithLogitsLoss":
                self.criterion_GAN = nn.BCEWithLogitsLoss(size_average=False).cuda(
                ) if self.gpu_mode else nn.BCEWithLogitsLoss(size_average=False)
            # elif self.loss_F == "Cross"

        # print('---------- Networks architecture -------------')
        # utils.print_network(self.model)
        # print('----------------------------------------------')

        # load dataset
        train_data_loader = self.load_dataset(
            dataset=self.train_dataset, is_train=True)
        test_data_loader = self.load_dataset(
            dataset=self.test_dataset[0], is_train=False)

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir)

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0
        avg_loss_D = []
        # test image
        test_lr, test_hr, test_bc, name = test_data_loader.dataset.__getitem__(
            2)

        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)
        test_bc = test_bc.unsqueeze(0)

        self.model.train()

        for epoch in range(self.num_epochs):
            # learning rate is decayed by a factor of 2 every 40 epochs
            if (epoch+1) % 40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2.0
                print('Learning rate decay: lr={}'.format(
                    self.optimizer.param_groups[0]['lr']))

            epoch_loss = 0
            epoch_loss_D = 0
            for iter, (lr, hr, bc_lr) in enumerate(train_data_loader):
                # input data (low resolution image)
                x_ = Variable(hr).cuda() if self.gpu_mode else Variable(hr)
                y_ = Variable(lr).cuda() if self.gpu_mode else Variable(hr)
                bc_y_ = Variable(bc_lr).cuda(
                ) if self.gpu_mode else Variable(hr)

                recon_image = self.model(y_)
                recon_image = recon_image + bc_y_
                # if 'Z' in self.model_loss:
                #     save_dir = os.path.join(self.save_dir, 'train_result')
                #     sr_img = recon_image[0].cpu().data
                #     save_img = utils.save_img(sr_img, epoch + 1,
                #                               save_dir=save_dir, is_training=True)
                #     self.zssr(save_img, self.model)
                # break
                loss_G = 0
                loss_D = 0
                loss_T = []
                if 'A' in self.model_loss:
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.valid = Variable(Tensor(np.ones(x_.size()[0]).reshape((x_.size()[0], 1))),
                                          requires_grad=False)
                    self.fake = Variable(Tensor(np.zeros(x_.size()[0]).reshape((x_.size()[0], 1))),
                                         requires_grad=False)
                    if self.gpu_mode:
                        self.valid.cuda()
                        self.fake.cuda()

                    self.optimizer_D.zero_grad()

                    # Loss of real and fake images
                    if self.loss_F == "BCEWithLogitsLoss":
                        loss_real = self.criterion_GAN(
                            self.discriminator(x_), self.valid)
                        loss_fake = self.criterion_GAN(
                            self.discriminator(recon_image.detach()), self.fake)
                        # Total loss
                        loss_D = (loss_real + loss_fake) / 2
                    elif self.loss_F == "MSE":
                        loss_real = self.mse_loss(
                            self.discriminator(x_), self.valid)
                        loss_fake = self.mse_loss(
                            self.discriminator(recon_image.detach()), self.fake)
                        # Total loss
                        loss_D = (loss_real + loss_fake) / 2
                    else:
                        loss_real = self.discriminator(x_)
                        loss_fake = self.discriminator(recon_image.detach())

                        loss_D = 0.5 * (torch.mean((loss_real - 1)**2) +
                                        torch.mean(loss_fake**2))
                    # # Total loss
                    # loss_D = (loss_real + loss_fake) / 2

                    loss_D.backward()
                    self.optimizer_D.step()
                    epoch_loss_D += loss_D.data[0]

                    if iter % self.D_period == 0:
                        # -----------------
                        #  Train Generator
                        # -----------------
                        self.optimizer.zero_grad()

                        loss_a, loss_output_m2, loss_output_m5, style_score, loss_G, loss_T = Loss.loss_op(self,
                                                                                                           self, recon_image, x_)

                        loss = (2*0.1*loss_output_m2) + \
                            (2*0.01*loss_output_m5) + \
                            loss_a*2 + style_score*1e-6
                        loss.backward()
                        self.optimizer.step()

                        # log
                        epoch_loss += loss.data[0]
                        utils.print_loss(self, epoch, len(train_data_loader), loss,
                                         style_score, loss_output_m2, loss_output_m5, iter, loss_a, loss_D, loss_G, loss_T)

                        # tensorboard logging
                        logger.scalar_summary('loss', loss.data[0], step + 1)
                        step += 1
                        del x_, y_, bc_y_
                        if self.gpu_mode:
                            torch.cuda.empty_cache()
                else:
                    # update network
                    self.optimizer.zero_grad()

                    loss_a, loss_output_m2, loss_output_m5, style_score, loss_G, loss_T = Loss.loss_op(self,
                                                                                                       self, recon_image, x_)
                    # loss = (2*0.1*loss_output_m2) + \
                    #     (2*0.01*loss_output_m5) + \
                    #     loss_a*1e-3 + style_score*1e-6
                    loss = (2*0.1*loss_output_m2) + \
                        (2*0.01*loss_output_m5) + \
                        loss_a*1e-3 + style_score

                    loss.backward()
                    self.optimizer.step()

                    # log
                    epoch_loss += loss.data[0]
                    #epoch_loss += loss
                    utils.print_loss(self, epoch, len(train_data_loader), loss,
                                     style_score, loss_output_m2, loss_output_m5, iter, loss_a, loss_D, loss_G, loss_T)

                    # tensorboard logging
                    logger.scalar_summary('loss', loss.data[0], step + 1)
                    step += 1
                    del x_, y_, bc_y_
                    if self.gpu_mode:
                        torch.cuda.empty_cache()

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

            # prediction
            y_ = Variable(test_lr).cuda(
            ) if self.gpu_mode else Variable(test_lr)
            bc = Variable(test_bc).cuda(
            ) if self.gpu_mode else Variable(test_bc)

            recon_img = self.model(y_)
            recon_img = recon_img+bc
            sr_img = recon_img[0].cpu().data

            # save result image
            save_dir = os.path.join(self.save_dir, 'train_result')
            save_img = utils.save_img(sr_img, epoch + 1,
                                      save_dir=save_dir, is_training=True)
            print('Result image at epoch %d is saved.' % (epoch + 1))

            # if 'Z' in self.model_loss:
            #     save_dir = os.path.join(self.save_dir, 'train_result')
            #     sr_img = recon_image[0].cpu().data
            #     save_img = utils.save_img(sr_img, epoch + 1,
            #                               save_dir=save_dir, is_training=True)
            #     self.zssr(save_img, self.model)
            utils.plot_loss([avg_loss], epoch, save_dir=save_dir)

            if 'A' in self.model_loss:
                avg_loss_D.append(epoch_loss_D / len(train_data_loader))
                utils.plot_loss([avg_loss_D], epoch, save_dir=save_dir+'/D')

            del y_, bc
            if self.gpu_mode:
                torch.cuda.empty_cache()

            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                utils.save_model(self, epoch + 1)
                test_v2.save_TrainingTest(self, epoch+1)
        # calculate psnrs
        if self.num_channels == 1:
            gt_img = test_hr[0][0].unsqueeze(0)
            lr_img = test_lr[0][0].unsqueeze(0)
            bc_img = test_bc[0][0].unsqueeze(0)
        else:
            gt_img = test_hr[0]
            lr_img = test_lr[0]
            bc_img = test_bc[0]

        bc_psnr = utils.PSNR(bc_img, gt_img)
        recon_psnr = utils.PSNR(sr_img, gt_img)

        # plot result images
        result_imgs = [gt_img, lr_img, bc_img, sr_img]
        psnrs = [None, None, bc_psnr, recon_psnr]
        utils.plot_test_result(
            result_imgs, psnrs, self.num_epochs, save_dir=save_dir, is_training=True)
        print('Training result image is saved.')

        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=save_dir)
        print('Training is finished.')

        # Save final trained parameters of model
        utils.save_model(self, epoch=None)

    def test(self):
        # networks
        self.model = Net(3, 64, 3, 1)
        # load model

        #utils.load_model(self, False, False)
        self.model.load_state_dict(torch.load('epoch0~200.pkl'))

        if self.gpu_mode:
            self.model.cuda()

        # load dataset
        for test_dataset in self.test_dataset:
            test_data_loader = self.load_dataset(
                dataset=test_dataset, is_train=False)

            # Test
            print('Test is started.')
            img_num = 0
            total_img_num = len(test_data_loader)
            self.model.eval()
            for lr, hr, bc, _ in test_data_loader:
                # input data (low resolution image)
                x_ = Variable(hr).cuda() if self.gpu_mode else Variable(hr)
                y_ = Variable(lr).cuda() if self.gpu_mode else Variable(hr)
                bc_ = Variable(bc).cuda(
                ) if self.gpu_mode else Variable(hr)

                # prediction
                recon_imgs = self.model(y_)
                recon_imgs += bc_
                for i, recon_img in enumerate(recon_imgs):
                    img_num += 1
                    sr_img = recon_img.cpu().data

                    # save result image
                    # save_dir = os.path.join(
                    #     self.save_dir, 'test_result_texture', test_dataset)
                    save_dir = os.path.join(
                        self.save_dir, 'test_result', test_dataset)
                    utils.save_img(sr_img, img_num, save_dir=save_dir)
                    # utils.save_img(x_, img_num, save_dir=os.path.join(
                    #     self.save_dir, 'test_HR', test_dataset))
                    # calculate psnrs
                    if self.num_channels == 1:
                        gt_img = hr[i][0].unsqueeze(0)
                        lr_img = lr[i][0].unsqueeze(0)
                        bc_img = bc[i][0].unsqueeze(0)
                    else:
                        gt_img = hr[i]
                        lr_img = lr[i]
                        bc_img = bc[i]

                    bc_psnr = utils.PSNR(bc_img, gt_img)
                    recon_psnr = utils.PSNR(sr_img, gt_img)

                    # plot result images
                    result_imgs = [gt_img, lr_img, bc_img, sr_img]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    utils.plot_test_result(
                        result_imgs, psnrs, img_num, save_dir=save_dir)

                    print(
                        'Test DB: %s, Saving result images...[%d/%d]' % (test_dataset, img_num, total_img_num))

            print('Test is finishied.')

    def zssr(self):
        self.model = Net(3, 64, 3, 1)
        # load model
        #utils.load_model(self, False, False)
        self.model.load_state_dict(torch.load('epoch0~200.pkl'))

        if self.gpu_mode:
            self.model.cuda()
        self.model.train()
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = False
        save_dir = os.path.join('z_result')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        avg_loss = []
        num_batch = 100000
        epoch_loss = 0
        learning_rate = 1e-5
        img = PIL.Image.open('SR_result_2.png')
        loss = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.model.parameters())[-1:], lr=learning_rate)
        sampler = DataSampler(img, self.scale_factor, self.crop_size)
        for iter, (hr, lr, bc) in enumerate(sampler.generate_data()):
            lr = Variable(lr).cuda()
            hr = Variable(hr).cuda()
            bc = Variable(bc).cuda()

            # img2 = lr.cpu().data[0, :, :, :]
            # oi2 = img2.numpy()
            # oi2[np.where(oi2 < 0)] = 0.0
            # oi2[np.where(oi2 > 1)] = 1.0
            # save_img2 = torch.from_numpy(oi2)
            # save_img2 = transforms.ToPILImage()(save_img2)
            # save_img2.show()

            # img3 = hr.cpu().data[0, :, :, :]
            # oi3 = img3.numpy()
            # oi3[np.where(oi3 < 0)] = 0.0
            # oi3[np.where(oi3 > 1)] = 1.0
            # save_img3 = torch.from_numpy(oi3)
            # save_img3 = transforms.ToPILImage()(save_img3)
            # save_img3.show()

            output = self.model(lr)
            output = output + bc
            error = loss(output, hr)
            if iter % 1000 == 0:
                img = output.cpu().data[0, :, :, :]
                oi = img.numpy()
                oi[np.where(oi < 0)] = 0.0
                oi[np.where(oi > 1)] = 1.0
                save_img = torch.from_numpy(oi)
                save_img = transforms.ToPILImage()(save_img)
                save_img.save(save_dir+"/train_%d.png" % iter)
            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                self.adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate))

            error.backward()
            optimizer.step()
            epoch_loss += error.data[0]

            # avg. loss per epoch
            avg_loss.append(epoch_loss / num_batch)
            if iter > num_batch:
                # Save final trained parameters of model
                utils.save_model(self, epoch=None)
                img2 = output.cpu().data[0, :, :, :]
                oi = img2.numpy()
                oi[np.where(oi < 0)] = 0.0
                oi[np.where(oi > 1)] = 1.0
                save_img = torch.from_numpy(oi)
                save_img = transforms.ToPILImage()(save_img)
                save_img.show()
                print('Done training.')

                utils.plot_loss([avg_loss], num_batch, save_dir=save_dir)

                break

        # load dataset
        for test_dataset in self.test_dataset:
            test_data_loader = self.load_dataset(
                dataset=test_dataset, is_train=False)

            # Test
            print('Test is started.')
            img_num = 0
            total_img_num = len(test_data_loader)
            self.model.eval()
            for lr, hr, bc, _ in test_data_loader:
                # input data (low resolution image)
                x_ = Variable(hr).cuda() if self.gpu_mode else Variable(hr)
                y_ = Variable(lr).cuda() if self.gpu_mode else Variable(hr)
                bc_ = Variable(bc).cuda(
                ) if self.gpu_mode else Variable(hr)

                # prediction
                recon_imgs = self.model(y_)
                recon_imgs += bc_
                for i, recon_img in enumerate(recon_imgs):
                    img_num += 1
                    sr_img = recon_img.cpu().data

                    # save result image
                    # save_dir = os.path.join(
                    #     self.save_dir, 'test_result_texture', test_dataset)
                    save_dir = os.path.join(
                        self.save_dir, 'z_result', test_dataset)
                    utils.save_img(sr_img, img_num, save_dir=save_dir)
                    # utils.save_img(x_, img_num, save_dir=os.path.join(
                    #     self.save_dir, 'test_HR', test_dataset))
                    # calculate psnrs
                    if self.num_channels == 1:
                        gt_img = hr[i][0].unsqueeze(0)
                        lr_img = lr[i][0].unsqueeze(0)
                        bc_img = bc[i][0].unsqueeze(0)
                    else:
                        gt_img = hr[i]
                        lr_img = lr[i]
                        bc_img = bc[i]

                    bc_psnr = utils.PSNR(bc_img, gt_img)
                    recon_psnr = utils.PSNR(sr_img, gt_img)

                    # plot result images
                    result_imgs = [gt_img, lr_img, bc_img, sr_img]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    utils.plot_test_result(
                        result_imgs, psnrs, img_num, save_dir=save_dir)

                    print(
                        'Test DB: %s, Saving result images...[%d/%d]' % (test_dataset, img_num, total_img_num))

            print('Test is finishied.')

    def adjust_learning_rate(self, optimizer, new_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
