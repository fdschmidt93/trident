from typing import Union

from torch.utils.data.dataset import Dataset

class PlainDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, list]):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]
