import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random

random.seed(777)
np.random.seed(777)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, x, y):
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            x = F.adjust_brightness(x, brightness_factor)
            y = F.adjust_brightness(y, brightness_factor)

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            x = F.adjust_contrast(x, contrast_factor)
            y = F.adjust_contrast(y, contrast_factor)

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            x = F.adjust_saturation(x, saturation_factor)
            y = F.adjust_saturation(y, saturation_factor)

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            x = F.adjust_hue(x, hue_factor)
            y = F.adjust_hue(y, hue_factor)

        x, y  = np.asarray(x), np.asanyarray(y)
        x, y  = x.clip(0, 255).astype(np.uint8), y.clip(0, 255).astype(np.uint8)
        x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y

class RandomColorChannel(object):
    def __call__(self, x, y):
        random_order = np.random.permutation(3)
        x, y = np.array(x), np.array(y)
        x, y = x[:,:,random_order], y[:,:,random_order]
        x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        shape = x.shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        x, y = x + (gaussian_noise*255), y + (gaussian_noise*255)
        x, y = x.clip(0, 255), y.clip(0, 255)
        x, y = x.astype(np.uint8), y.astype(np.uint8)
        x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, x, y):
        x = (np.array(x)/255-self.mean)/self.std
        y = (np.array(y)/255-self.mean)/self.std
        return x, y

class CenterCrop(object):
    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, x, y):
        input_size_h, input_size_w, _ = x[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))
        x = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in x]
        y = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in y]
        return x, y

class RandomCrop(object):
    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]
    def __call__(self, x, y):
        input_size_h, input_size_w, _ = x.shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)
        x  = x[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        y = y[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        return x, y

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""
    def __call__(self, x, y):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            x = np.copy(np.fliplr(x))
            y = np.copy(np.fliplr(y))
            x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = np.copy(np.flipud(x))
            y = np.copy(np.flipud(y))
            x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, x, y):
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        x  = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y