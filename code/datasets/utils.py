import torch

def calculate_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for batch in dataloader:
        rgbs = batch['rgb']
        channels_sum += torch.mean(rgbs, dim=[0,2,3])
        channels_squared_sum += torch.mean(rgbs**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
