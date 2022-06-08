import os.path
import numpy as np
from . import custom_transforms
from .base_dataset import BaseDataset
from .image_handler import make_dataset_image
from PIL import Image


class UDCDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.phase = opt.phase
        self.root = opt.dataroot if opt.phase == 'train' else opt.testroot
        self.dir_img = os.path.join(self.root)
        self.imgX_paths, self.imgY_paths = make_dataset_image(self.dir_img)
        if opt.phase == 'train':
            if opt.dataset == "SYNTH":
                self.transform_list = [#custom_transforms.RandomGaussianNoise([0, 1e-3]),
                                       custom_transforms.ToTensor()]
            else:
                self.transform_list = [custom_transforms.RandomVerticalFlip(),
                                       custom_transforms.RandomHorizontalFlip(),
                                       custom_transforms.RandomGaussianNoise([0, 1e-3]),
                                       custom_transforms.Normalize(0, 1),
                                       custom_transforms.ToTensor()
                                       ]
        else:
            if opt.dataset == "SYNTH":
                self.transform_list = [custom_transforms.ToTensor()
                                       ]
            else:
                self.transform_list = [custom_transforms.Normalize(0, 1),
                                       custom_transforms.ToTensor()
                                       ]
        self.transform = custom_transforms.Compose(self.transform_list)

    def _tonemap(self, x, type='simple'):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        else:
            raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

    def __getitem__(self, index):
        imgX_path = self.imgX_paths[index]
        imgY_path = self.imgY_paths[index]

        if self.opt.dataset == "SYNTH":
            imgX = np.load(imgX_path)
            imgY = np.load(imgY_path)
            imgX = self._tonemap(imgX)
            imgY = self._tonemap(imgY)
        else:
            imgX = Image.open(imgX_path).convert('RGB')
            imgY = Image.open(imgY_path).convert('RGB')
        imgX, imgY = self.transform(imgX, imgY)

        X = imgX
        Y = imgY

        return {'X': X, 'Y': Y, 'img_path': imgX_path}

    def __len__(self):
        return len(self.imgX_paths)

    def name(self):
        return 'udc_datset'
