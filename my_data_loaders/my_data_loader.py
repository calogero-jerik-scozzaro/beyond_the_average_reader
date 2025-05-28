from torch.utils.data import DataLoader

class MyDataLoader(DataLoader):
    def __init__(self, dataset, config, shuffle=True):
        super().__init__(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=shuffle, 
            drop_last=False
        )