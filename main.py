# -*- coding: UTF-8 -*-
import torch
import os
import argparse
from EnhanceNet import EnhanceNet
import time
"""parsing and configuration"""


def parse_args():
    desc = "PyTorch implementation of SR collections"
    train_dataset = "Set14"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='EnhanceNet')
    parser.add_argument('--model_loss', type=str, default='PAT')
    parser.add_argument('--D_period', type=int, default=3)
    parser.add_argument('--data_dir', type=str,
                        default='/home/home/Desktop/Dataset')
    parser.add_argument('--train_dataset', type=list, default=[train_dataset], choices=['train2014', 'train2014 1960_hr', 'train2017-39907', 'train2014_2000'],
                        help='The name of training dataset')
    parser.add_argument('--test_dataset', type=list, default=['Set5', 'Set14'], choices=['text', 'Set5', 'Set14', 'people', 'food'],
                        help='The name of test dataset')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=100,  # 調整
                        help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='The number of channels to super-resolve')
    parser.add_argument('--scale_factor', type=int,
                        default=4, help='Size of scale factor')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='The number of epochs to run')
    parser.add_argument('--previous_epochs', type=int, default=0,
                        help='The number of previous epochs to run')
    parser.add_argument('--save_epochs', type=int, default=1,
                        help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=1, help='testing batch size')
    parser.add_argument('--save_dir', type=str, default='Result_'+train_dataset,
                        help='Directory name to save the results')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--patchloss', type=bool, default=True)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--loss_F', type=str, default='BCEWithLogitsLoss',
                        choices=["BCEWithLogitsLoss", "MSE", "BCE", "CrossEntropyLoss"])

    #parser.add_argument('--gpu_mode', type=bool, default=False)

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""


def main():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model

    if args.model_name == 'EnhanceNet':
        net = EnhanceNet(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)

    net.train()

    # test
    net.test()
    # net.test_single('getchu_full.jpg')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print("Time taken: ", elapsed, "seconds.")
