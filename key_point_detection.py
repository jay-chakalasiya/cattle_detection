import os
import pickle
import torch
from torch.utils.data import Dataset

from config import DATA_PREP_PATH, DATA_PREP_FILES, BOX_MARGIN


class CattleDataset(Dataset):
    def __init__(self):
        self.X, self.Y_key_points = self.load_data()
        self.dataset_length = len(self.X)

    def __len__(self):
        return self.dataset_length
        
    def __getitem__(self, index):
        x = self.X[index]
        y_box = [self.make_box(x.shape)]
        y_label = [1]
        y_key_points = self.Y_key_points[index]
        return {'image': torch.tensor(x/255, dtype=torch.float), 
                'boxes': torch.tensor(y_box, dtype=torch.float),
                'labels': torch.tensor(y_label, dtype=torch.int64),
                'keypoints': torch.tensor(y_key_points, dtype=torch.float)}

    def make_box(self, image_shape):
        return [BOX_MARGIN, BOX_MARGIN, image_shape[2]-BOX_MARGIN, image_shape[1]-BOX_MARGIN]

    def load_data(self):
        f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['key_point_data']), 'rb')
        data = pickle.load(f)
        images, key_point_labels = data['images'], data['labels']
        return images, key_point_labels


def collate_fn(*batch):
    return list(zip(batch))[0][0]