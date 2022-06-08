import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable


###############################################################################
# Functions
###############################################################################

def init_loss(opt):
    content_loss = PSNRLoss()
    perceptual_loss = PerceptualLoss()
    perceptual_loss.initialize(nn.MSELoss())
    return content_loss, perceptual_loss

class HueLoss(nn.Module):
    def __init__(self):
        super(HueLoss, self).__init__()

    def get_loss(self, pred, target):
        N, C, H, W = pred.shape
        pred_r = pred[:, 0, :, :]       # N, 1, H, W
        pred_g = pred[:, 1, :, :]       # N, 1, H, W
        pred_b = pred[:, 2, :, :]       # N, 1, H, W
        pred_hue = torch.atan2((pred_g - pred_b) * torch.sqrt(3), (pred_r - pred_g - pred_b) * 2) # N, 1, H, W

        target_r = target[:, 0, :, :]  # N, 1, H, W
        target_g = target[:, 1, :, :]  # N, 1, H, W
        target_b = target[:, 2, :, :]  # N, 1, H, W
        target_hue = torch.atan2((target_g - target_b) * torch.sqrt(3), (target_r - target_g - target_b) * 2)  # N, 1, H, W

        diff = ((pred_hue - target_hue) ** 2).mean()
        return diff

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def get_loss(self, pred, target):
        N, C, H, W = pred.shape
        mean_pred = pred.view(N, C, -1).mean(2, keepdim=True).unsqueeze(3)      # N, C, 1, 1
        contrast_pred = (pred - mean_pred) ** 2                                 # N, C, H, W
        contrast_pred = contrast_pred.mean(dim=(1, 2, 3))                       # N

        mean_target = target.view(N, C, -1).mean(2, keepdim=True).unsqueeze(3)
        contrast_target = (target - mean_target) ** 2
        contrast_target = contrast_target.mean(dim=(1, 2, 3))

        diff = ((contrast_pred - contrast_target) ** 2).mean()
        return diff

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def get_loss(self, pred, target):
        pred = pred * 255.
        target = target * 255.
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def get_loss(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def get_loss(self, x, y):
        loss = self.loss.get_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class TVLoss():
    """
    Total variation loss.
    """
    def initialize(self):
        pass
    def get_loss(self, flow):
        bsize, chan, height, width = flow.size()
        tvhs = []
        tvws = []
        for h in range(height-1):
            dy = torch.abs(flow[:,:,h+1,:] - flow[:,:,h,:])
            tvh = torch.norm(dy, 1)
            tvhs.append(tvh)
        for w in range(width-1):
            dx = torch.abs(flow[:,:,:,w+1] - flow[:,:,:,w])
            tvw = torch.norm(dx, 1)
            tvws.append(tvw)

        return sum(tvhs + tvws) / (height + width)

class ClassificationLoss():
    def initialize(self, loss):
        self.criterion = loss
    def get_loss(self, label_h, label_y):
        return self.criterion(label_h, label_y)


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss
    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        conv_5_4_layer = 33
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_5_4_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)

class DiscLoss():
    def name(self):
        return 'DiscLoss'

    def initialize(self):
        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, net, fake):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fake)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, real, fake):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fake.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(real)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def initialize(self):
        DiscLoss.initialize(self)
        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, net, fake):
        return DiscLoss.get_g_loss(self, net, fake)

    def get_loss(self, net, real, fake):
        return DiscLoss.get_loss(self, net, real, fake)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self):
        DiscLossLS.initialize(self)
        self.LAMBDA = 10

    def get_g_loss(self, net, fake):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fake)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, real, fake):
        self.D_fake = net.forward(fake.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(real)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, real.data, fake.data)
        return self.loss_D + gradient_penalty

