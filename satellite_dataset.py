from torch.utils.data import DataSet


class SatelliteDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.train = df.iloc[~df.is_valid].reset_index()
        self.valid = df.iloc[df.is_valid].reset_index()
        self.transform = transforms
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        elif self.mode == 'val':
            return len(self.valid)

    def __getitem__(self, idx):
        if self.mode == "train":
            image_fname = data_dir/ self.train.name[idx]
            mask_fname = self.train.label[idx]
