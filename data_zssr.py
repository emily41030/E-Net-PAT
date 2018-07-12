import PIL
import numpy as np
import sys
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import cv2
from source_target_transforms import *
import os


class DataSampler:
    def __init__(self, img, sr_factor, crop_size):
        self.img = img
        self.sr_factor = sr_factor
        self.pairs = self.create_hr_lr_pairs()
        sizes = np.float32([x[0].size[0]*x[0].size[1] / float(img.size[0]*img.size[1])
                            for x in self.pairs])
        self.pair_probabilities = sizes / np.sum(sizes)

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            # RandomCrop(crop_size),
            ToTensor()])

    def create_hr_lr_pairs(self):
        smaller_side = min(self.img.size[0: 2])
        larger_side = max(self.img.size[0: 2])

        factors = []
        for i in range(smaller_side//5, smaller_side+1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side)/smaller_side
            downsampled_larger_side = round(larger_side*zoom)
            if downsampled_smaller_side % self.sr_factor == 0 and \
                    downsampled_larger_side % self.sr_factor == 0:
                factors.append(zoom)

        pairs = []
        count = 0
        for zoom in factors:
            x = int(self.img.size[0]*zoom)
            y = int(self.img.size[1]*zoom)
            if x % 4 != 0:
                x = x+4-(x % 4)
            if y % 4 != 0:
                y = y+4-(y % 4)
            hr = self.img.resize((x, y),
                                 resample=PIL.Image.BICUBIC)

            lr = hr.resize((int(x/self.sr_factor),
                            int(y/self.sr_factor)),
                           resample=PIL.Image.BICUBIC)

            # lr.save(os.path.join("mulLR_image")+'/lr_'+str(count)+'.png')
            bc = lr.resize(hr.size, resample=PIL.Image.BICUBIC)
            count += 1
            pairs.append((hr, lr, bc))

        return pairs

    def generate_data(self):
        while True:
            hr, lr, bc = random.choices(
                self.pairs, weights=self.pair_probabilities, k=1)[0]
            # lr.show()
            # hr.show()
            # bc.show()
            hr_tensor, lr_tensor, bc_tensor = self.transform((hr, lr, bc))
            hr_tensor = torch.unsqueeze(hr_tensor, 0)
            lr_tensor = torch.unsqueeze(lr_tensor, 0)
            bc_tensor = torch.unsqueeze(bc_tensor, 0)
            yield hr_tensor, lr_tensor, bc_tensor


if __name__ == '__main__':
    img = PIL.Image.open(sys.argv[1])
    sampler = DataSampler(img, 2)
    for x in sampler.generate_data():
        hr, lr = x
        hr = hr.numpy().transpose((1, 2, 0))
        lr = lr.numpy().transpose((1, 2, 0))
