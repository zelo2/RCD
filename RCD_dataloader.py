import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import KFold


class as_RCD_Dataset(Dataset):
    def __init__(self, data, q):
        super(as_GD_DINA_Dataset, self).__init__()
        self.data = data
        self.q = q

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        stu_id = self.data[index][0]
        ques_id = self.data[index][1]
        q_vector = self.q[self.data[index][1]]
        label = self.data[index][2]
        return stu_id, ques_id, q_vector, label



