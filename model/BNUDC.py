from util.util import tensor2im
from collections import OrderedDict
from .base_model import BaseModel
from .network import BNUDCnet, print_network
from .losses import init_loss
from warmup_scheduler import GradualWarmupScheduler
import torch
import numpy as np


class Model(BaseModel):
    def name(self):
        return 'Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.net = BNUDCnet().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=opt.gpu_ids)

        if self.isTrain:
            # optimizers
            self.optimizer= torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=1e-8)
            self.warmup_epochs = 3
            self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, opt.niter - self.warmup_epochs, eta_min=opt.lr_min)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine)
            self.scheduler.step()
            # loss
            self.content_loss, self.perceptual_loss = init_loss(opt)

        if not self.isTrain or opt.continue_train or opt.phase == 'test':
            self.load_network(self.net, 'bnudc', opt.which_epoch)
            print('Network was successfully loaded!!')

        if self.isTrain:
            if opt.continue_train:
                start_epoch = int(opt.which_epoch) + 1
                for i in range(1, start_epoch):
                    self.scheduler.step()
            print('Lr is %f now' % self.scheduler.get_lr()[0])

    def set_input(self, input):
        self.X_rgb = input['X'].cuda()
        self.Y_rgb = input['Y'].cuda()
        self.path = input['img_path']

    def forward(self):
        self.x_high, self.x_low, self.x_hat = self.net(self.Y_rgb)

    def backward(self):
        self.loss_recon = self.content_loss.get_loss(self.x_hat, self.X_rgb)
        self.loss_total = self.loss_recon
        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        return self.path

    def get_current_errors(self):
        return OrderedDict([('loss_recon', self.loss_recon.data.cpu().numpy()),
                            ('loss_total', self.loss_total.data.cpu().numpy())
                            ])

    def get_eval_data(self):
        x = torch.clamp(self.X_rgb[0].data, 0, 1).cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        y = torch.clamp(self.Y_rgb[0].data, 0, 1).cpu().numpy()
        y = np.transpose(y, (1, 2, 0))
        x_hat = torch.clamp(self.x_hat[0].data, 0, 1).cpu().numpy()
        x_hat = np.transpose(x_hat, (1, 2, 0))
        return OrderedDict([('x', x), ('y', y), ('x_hat', x_hat)
                            ])

    def get_tensor_raw_data(self):
        x = self.X_rgb.cpu()
        y = self.Y_rgb.cpu()
        x_hat = torch.clamp(self.x_hat.cpu(), 0, 1)
        return OrderedDict(
            [('x', x), ('y', y), ('x_hat', x_hat)
             ])

    def get_current_visuals(self):
        is_scale = False
        x = tensor2im(self.X_rgb.data, is_scale=is_scale)
        y = tensor2im(self.Y_rgb.data, is_scale=is_scale)
        x_high = tensor2im(torch.clamp(self.x_high.data, 0, 1), is_scale=is_scale)
        x_low = tensor2im(torch.clamp(self.x_low.data, 0, 1), is_scale=is_scale)
        x_hat = tensor2im(torch.clamp(self.x_hat.data, 0, 1), is_scale=is_scale)
        return OrderedDict(
            [('x', x), ('y', y),
             ('x_high', x_high), ('x_low', x_low),
             ('x_high', x_high), ('x_low', x_low),
             ('x_hat', x_hat)])

    def save(self, label):
        self.save_network(self.net, 'bnudc', label, self.gpu_ids)

    def warmup_scheduler(self):
        self.scheduler.step()
        print('LR is updated : %f' % self.scheduler.get_lr()[0])




