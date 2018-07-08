from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from dataset import *


def get_training_set(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dir = []
    for dataset in datasets:
        if dataset == 'DIV2K':
            train_dir.append(
                join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X2'))
        elif dataset == 'train2017':
            train_dir.append(join(data_dir, dataset))
        else:
            train_dir.append(join(data_dir, dataset))

    return TrainDatasetFromFolder(train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)


def get_test_set(data_dir, dataset, scale_factor, is_gray=False):
    if dataset == 'DIV2K':
        test_dir = join(data_dir, dataset, 'DIV2K_test_LR_bicubic/test/hr')

    elif dataset == 'people':
        test_dir = join(data_dir, dataset, 'LR')
    elif dataset == 'food':
        test_dir = join(data_dir, dataset, 'LR')
    else:
        test_dir = join(data_dir, dataset)

    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)
