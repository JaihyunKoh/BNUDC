from .dataset_data_loader import DatasetDataLoader


def CreateDataLoader(opt):
    data_loader = DatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
