from base_networks import *
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import numpy as np


def save_TrainingTest(self, epoch):
    # load dataset
    for test_dataset in self.test_dataset:
        test_data_loader = self.load_dataset(
            dataset=test_dataset, is_train=False)

        # Test
        print('Test is started.')
        img_num = 0
        total_img_num = len(test_data_loader)
        self.model.eval()

        for lr, hr, bc, name in test_data_loader:
            # input data (low resolution image)
            if self.num_channels == 1:
                y_ = Variable(lr[:, 0].unsqueeze(1))
            else:
                y_ = Variable(lr)
                bc_ = Variable(bc)
                x_ = Variable(hr)
            if self.gpu_mode:
                y_ = y_.cuda()
                bc_ = bc_.cuda()
                x_ = x_.cuda()
            # prediction
            recon_imgs = self.model(y_)
            recon_imgs += bc_
            for i, recon_img in enumerate(recon_imgs):
                sr_img = recon_img.cpu().data

                # save result image
                save_dir = os.path.join(
                    self.save_dir, 'test_result/train_epoch', test_dataset)
                # save_img(sr_img, img_num, is_training=True, save_dir=save_dir, epoch)
            ###########################################
            # img.clamp(0, 1)
                if list(sr_img.shape)[0] == 3:
                    save_img = sr_img*255.0
                    save_img = save_img.clamp(0, 255).numpy(
                    ).transpose(1, 2, 0).astype(np.uint8)
                    # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
                else:
                    save_img = sr_img.squeeze().clamp(0, 1).numpy()

                # save img
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_fn = save_dir + \
                    '/'+str(name[img_num])+'_result_epoch_{:d}'.format(
                        epoch) + '.png'

                plt.imsave(save_fn, save_img)
                ########################################
                # calculate psnrs
                gt_img = hr[i]
                lr_img = lr[i]
                bc_img = bc[i]

                # 原始輸入圖存檔
                if epoch == self.save_epochs:
                    save_fn2 = save_dir + \
                        '/'+str(name[img_num])+'_lr.png'
                    if list(lr_img.shape)[0] == 3:
                        save_img = lr_img*255.0
                        save_img = save_img.clamp(0, 255).numpy(
                        ).transpose(1, 2, 0).astype(np.uint8)
                    plt.imsave(save_fn2, save_img)

                bc_psnr = utils.PSNR(bc_img, gt_img)
                recon_psnr = utils.PSNR(sr_img, gt_img)

                # plot result images
                result_imgs = [gt_img, lr_img, bc_img, sr_img]
                psnrs = [None, None, bc_psnr, recon_psnr]
                utils.plot_test_result(
                    result_imgs, psnrs, img_num, save_dir=save_dir, is_training=True)
                del x_, y_, bc_
                if self.gpu_mode:
                    torch.cuda.empty_cache()
                img_num += 1
                print(
                    'Test DB: %s, Saving result images...[%d/%d]' % (test_dataset, img_num, total_img_num))

        print('Test is finishied.')
