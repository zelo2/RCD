import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold
from model import NCDF
from model import pmf_cd
from model import SGD_PMF
from model import tradition_CF
from snack import DINA_GD_dataloader
from psy import EmDina, MlDina
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from snack import assistment_process
import pandas as pd
from model import RCD


if __name__ == '__main__':
    as_2009 = True
    if as_2009:
        data = pd.read_csv(
            '2009_skill_builder_data_corrected/skill_builder_data_corrected1.csv',
            encoding="utf-8")
        device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
        score, q, indicator, train_data, test_data = assistment_process.as_data_process(data, percent=0.8)
        label_index = np.array(np.where(indicator == 0)).T

        '''Data Description'''
        stu_num = len(score)
        item_num = len(score[0])
        skill_num = len(q[0])
        logs_num = np.sum(1 - np.isnan(score))
        print('Logs Number:', logs_num)
        print('Student Number:', stu_num)
        print('Item Number:', item_num)
        print("SKill Number:", skill_num)
        sub_id = 1000000
        sub_threshold = 0.1
        k_value = 10

        '''RCD-Ration-aware Cognitive Diagnosis'''
        train_data = DINA_GD_dataloader.as_GD_DINA_Dataset(train_data, q)
        train_dataloder = DataLoader(dataset=train_data, batch_size=256, shuffle=True)

        net = RCD.RCDNet(stu_num, item_num, skill_num, device)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        loss_function = nn.BCELoss()
        cdm_skill = torch.zeros(stu_num, skill_num).to(device)
        '''Start Training Model'''
        for epoch in range(60):
            running_loss = 0
            for i, (stu_id, ques_id, q_vector, label) in enumerate(train_dataloder, 0):
                '''Warning: Embedding parameters should be longtensor'''
                # label = label.reshape([-1, 1])
                stu_id = torch.LongTensor(stu_id)
                ques_id = torch.LongTensor(ques_id)
                q_vector = q_vector.float()
                stu_id, ques_id, q_vector, label = stu_id.to(device), ques_id.to(device), q_vector.to(
                    device), label.to(device)
                optimizer.zero_grad()

                output, cdm_skill = net.forward(stu_id, ques_id, q, indicator)
                label = label.float()
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("epoch:", epoch, "loss:", running_loss)

            with torch.no_grad():
                precision_easy = 0  # 精确率
                recall_easy = 0  # 召回率

                precision_hard = 0  # 精确率
                recall_hard = 0  # 召回率

                TP_easy = 0
                FP_easy = 0
                FN_easy = 0

                TP_hard = 0
                FP_hard = 0
                FN_hard = 0

                acc = 0
                rmse = 0

                net.eval()
                test_data = DINA_GD_dataloader.as_GD_DINA_Dataset(test_data, q)
                test_dataloder = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
                for i, (stu_id, ques_id, q_vector, label) in enumerate(test_dataloder, 0):
                    '''Warning: Embedding parameters should be longtensor'''
                    stu_id = torch.LongTensor(stu_id)
                    ques_id = torch.LongTensor(ques_id)
                    q_vector = q_vector.float()
                    stu_id, ques_id, q_vector, label = stu_id.to(device), ques_id.to(device), q_vector.to(
                        device), label.to(device)

                    output, cdm_skill = net.forward(stu_id, ques_id, q, indicator)
                    sub_mark = 1000000

                    # acc
                    if (output.item() >= 0.5 and label.item() == 1) or (output.item < 0.5 and label.item() == 0):
                        acc += 1

                    # rmse
                    rmse += (output.item() - label.item()) ** 2

                    if output >= 0.5 and ques_id < sub_mark:
                        output = 1
                    elif output < 0.5 and ques_id < sub_mark:
                        output = 0
                    else:
                        output = output.item()

                    if label.item() >= 0.5:  # easy exercise
                        FN_easy += 1
                        if np.abs((output - label.item())) <= 0.1:
                            TP_easy += 1
                        else:
                            FP_easy += 1
                    else:
                        FN_hard += 1
                        if np.abs((output - label.item())) <= 0.1:
                            TP_hard += 1
                        else:
                            FP_hard += 1

                precision_easy = TP_easy / (TP_easy + FP_easy)
                precision_hard = TP_hard / (TP_hard + FP_hard)
                recall_easy = TP_easy / (TP_easy + FN_easy)
                recall_hard = TP_hard / (TP_hard + FN_hard)
                acc = acc / len(test_data)
                rmse = torch.sqrt(rmse / len(test_data))
                print("Preciseion Easy:", precision_easy)
                print("Preciseion Hard:", precision_hard)
                print("Recall Easy:", recall_easy)
                print("Recall Hard:", recall_hard)
                print("F1 Easy:", (2 * precision_easy * recall_easy) / (precision_easy + recall_easy))
                print("F1 Hard:", (2 * precision_hard * recall_hard) / (precision_hard + recall_hard))
                print("Accuracy:", acc)
                print("RMSE:", rmse)







