import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import KFold

class GD_DINA_Dataset(Dataset):
    def __init__(self, score, q, ques_index):
        super(GD_DINA_Dataset, self).__init__()
        self.score = score.reshape(-1)  # 得分矩阵
        self.q = q  # 知识点矩阵
        self.question_num = len(score[0])
        self.ques_index = ques_index

    def __len__(self):
        return len(self.score)

    def __getitem__(self, index):
        stu_id = int(index / self.question_num)
        ques_id = self.ques_index[stu_id][(index+self.question_num) % self.question_num]
        q_vector = self.q[ques_id]
        label = self.score[index]
        return stu_id, ques_id, q_vector, label

class as_GD_DINA_Dataset(Dataset):
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



