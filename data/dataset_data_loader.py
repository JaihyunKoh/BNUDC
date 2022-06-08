import torch.utils.data
from .base_data_loader import BaseDataLoader
from .udc_dataset import UDCDataset




def CreateDataset(opt):
    dataset = UDCDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class DatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'DatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=(opt.phase == 'train'),
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
