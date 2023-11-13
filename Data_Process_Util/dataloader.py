import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class AlloyDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path, engine='openpyxl')
        self.data = self.data.iloc[0:699]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        compositions = torch.tensor(self.data.iloc[idx, 1:7].values.astype(float), dtype=torch.float32)
        properties = torch.tensor(self.data.iloc[idx, 7:], dtype=torch.float32)
        return compositions, properties

def create_dataloader(file_path, batch_size, train_ratio=1.0):
    dataset = AlloyDataset(file_path)
    train_len = int(train_ratio * len(dataset))
    valid_len = len(dataset) - train_len
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


# if __name__ == '__main__':
#     cuda = True if torch.cuda.is_available() else False
#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor # type: ignore
#     LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor # type: ignore

#     file_path = '../Data_Warehouse/data.xlsx'
#     train_dataloader, valid_dataloader = create_dataloader(file_path, batch_size=50)
#     for i, (compositions, properties) in enumerate(train_dataloader):
#         batch_size = compositions.shape[0]
#         real_comps = compositions.type(Tensor)
#         labels = properties.type(LongTensor)
#         latent_code = Tensor(np.random.normal(0, 1, (batch_size, 16)))
#         print(real_comps.data)
#         break