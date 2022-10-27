import threading

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from snack import assistment_process
import pandas as pd
import scipy.sparse as sp
import tqdm
import os

class RCDNet(nn.Module):
    def __init__(self, user_num, item_num, skill_num, device, layer_num=1):
        super(RCDNet, self).__init__()

        self.device = device
        self.user_num = user_num
        self.item_num = item_num
        self.skill_num = skill_num
        self.layer_num = layer_num  # layer of GNN
        self.embed_size = self.skill_num

        # Embedding Layer
        self.user_layer = nn.Embedding(self.user_num, self.embed_size)
        self.item_layer = nn.Embedding(self.item_num, self.embed_size)
        self.skill_layer = nn.Embedding(self.skill_num, self.embed_size)

        # Liner Transformation Layer
        '''Stu Fusion'''
        self.stu_item_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)
        self.stu_fusion_transformation_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)

        '''Item Fusion'''
        self.item_stu_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)
        self.item_skill_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)
        self.item_fusion_transformation_layer_4_stu = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.item_fusion_transformation_layer_4_skill = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.map_level_item_stu_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)
        self.map_level_item_skill_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)

        '''Skill Fusion'''
        self.skill_item_attention_layer = nn.Linear(self.embed_size * 2, 1, bias=False)
        self.skill_fusion_transformation_layer_4_item = nn.Linear(self.embed_size, self.embed_size, bias=False)

        '''Prediction Layer'''
        self.fuse_stu_skill_layer = nn.Linear(self.embed_size * 2, self.embed_size)
        self.fuse_item_skill_layer = nn.Linear(self.embed_size * 2, self.embed_size)
        self.performance_prediction_layer = nn.Linear(self.embed_size, 1)


    def adj_matrix_creation(self, source='as0910'):
        if source == 'as0910':
            data = pd.read_csv(
                '../2009_skill_builder_data_corrected/skill_builder_data_corrected1.csv',
                encoding="utf-8")
        # indicator- 0:test 1:train
        score, q, indicator, train_data, test_data = assistment_process.as_data_process(data, percent=0.8)

        item_skill_adj_matrix = sp.dok_matrix((self.item_num, self.skill_num), dtype=np.float32)
        # item_skill_adj_matrix = item_skill_adj_matrix.tolil()  # equal to q-matrix

        stu_item_adj_matrix = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        # stu_item_adj_matrix = stu_item_adj_matrix.tolil()

        for i in train_data:
            # skill-exercise
            item_sample = i[1]
            q_vector = q[item_sample]
            skill_index = np.where(q_vector == 1)[0]
            for j in skill_index:
                item_skill_adj_matrix[item_sample, j] = 1  # [item_sample][j]  is Wrong!
            # user-item
            user_sample = i[0]
            stu_item_adj_matrix[user_sample, item_sample] = 1

        return stu_item_adj_matrix, item_skill_adj_matrix

    def norm_adj_single(adj):
        # D^-(1/2) * A * D^-(1/2)
        row_sum = np.array(adj.sum(1))  # sum each row
        d_half = np.power(row_sum, -0.5).flatten()
        d_half[np.isinf(d_half)] = 0.
        d_half_matrix = sp.diags(d_half)

        result = d_half_matrix.dot(adj).dot(d_half_matrix)
        return result.tocoo()

    def convert_coo_matirix_2_sp_tensor(self, X, tensor_type='float'):
        coo = X.tocoo()
        # coo matrix--((data, (row, column)), shape)
        # data:矩阵中的数据， row, column表示这个数据在哪一行哪一列
        i = torch.LongTensor([coo.row, coo.col])  # [row, column]
        v = torch.from_numpy(coo.data).float()  # data
        if tensor_type == 'float':
            return torch.sparse.FloatTensor(i, v, coo.shape)
        elif tensor_type == 'long':
            return torch.sparse.LongTensor(i, v, coo.shape)

    def forward(self, user, item, q, indicator):
        batch_size = len(user)
        user_embed = self.user_layer.weight  # user_num * embed
        item_embed = self.item_layer.weight  # item_num * embed
        skill_embed = self.skill_layer.weight  # skill_num * embed
        final_user_embeddings = torch.zeros_like(self.user_layer.weight).to(self.device)
        final_item_embeddings = torch.zeros_like(self.item_layer.weight).to(self.device)
        final_skill_embeddings = torch.zeros_like(self.skill_layer.weight).to(self.device)
        for i in range(self.layer_num):
            '''-------Stu fusion-------'''
            print("-------Student Fusion-------")
            stu_item_attention_score = torch.sparse_coo_tensor(size=(self.user_num, self.item_num)).to(self.device)
            for j in range(self.user_num):
                item_index = np.where(indicator[j] == 1)[0]
                one_user_embed = user_embed[j].repeat(len(item_index), 1)  # [item, emd]
                one_stu_item_attention_input = torch.cat([self.stu_fusion_transformation_layer(one_user_embed),
                                                          self.stu_fusion_transformation_layer(item_embed[item_index])],
                                                         dim=1)  # [item, emd*2]

                one_stu_item_attention_score = self.stu_item_attention_layer(one_stu_item_attention_input)  # [item, 1]
                one_stu_item_attention_score = one_stu_item_attention_score.squeeze(-1)  # [item]
                one_stu_item_attention_score = F.softmax(one_stu_item_attention_score)  # [item]

                stu_item_attention_score = stu_item_attention_score.to_dense()
                stu_item_attention_score[j, item_index] = one_stu_item_attention_score
                stu_item_attention_score = stu_item_attention_score.to_sparse()

            transformed_item_embeddings = self.stu_fusion_transformation_layer(item_embed)
            # [user, item] mm [item, emb] -> [user, emb]
            user_embeddings = user_embed + torch.sparse.mm(stu_item_attention_score, transformed_item_embeddings)
            final_user_embeddings = final_user_embeddings + user_embeddings

            '''-------Item fusion-------'''
            print("-------Exercise Fusion-------")
            item_stu_attention_score = torch.sparse_coo_tensor(size=(self.item_num, self.user_num)).to(self.device)
            item_skill_attention_score = torch.sparse_coo_tensor(size=(self.item_num, self.skill_num)).to(self.device)
            for j in range(self.item_num):
                # attention score for fusing stu
                user_index = np.where(indicator[:, j] == 1)[0]
                one_item_embed = item_embed[j].repeat(len(user_index), 1)  # [user,emd]

                one_item_stu_attention_input = torch.cat([self.item_fusion_transformation_layer_4_stu(one_item_embed),
                                                          self.item_fusion_transformation_layer_4_stu(user_embed[user_index])], dim=1)  # [user, emd]
                one_stu_item_attention_score = self.item_stu_attention_layer(one_item_stu_attention_input)
                one_stu_item_attention_score = one_stu_item_attention_score.squeeze(-1)  # [item]
                one_stu_item_attention_score = F.softmax(one_stu_item_attention_score)

                item_stu_attention_score = item_stu_attention_score.to_dense()
                item_stu_attention_score[j, user_index] = one_stu_item_attention_score
                item_stu_attention_score = item_stu_attention_score.to_sparse()

                # attention score for fusing skill
                skill_index = np.where(q[j] == 1)[0]
                one_item_embed = item_embed[j].repeat(len(skill_index), 1)  # [skill, emd]

                one_item_skill_attention_input = torch.cat([self.item_fusion_transformation_layer_4_skill(one_item_embed),
                                                            self.item_fusion_transformation_layer_4_skill(skill_embed[skill_index])], dim=1)
                one_item_skill_attention_score = self.item_skill_attention_layer(one_item_skill_attention_input)  # [skill, 1]
                one_item_skill_attention_score = one_item_skill_attention_score.squeeze(-1)
                one_item_skill_attention_score = F.softmax(one_item_skill_attention_score)

                item_skill_attention_score = item_skill_attention_score.to_dense()
                item_skill_attention_score[j, skill_index] = one_item_skill_attention_score
                item_skill_attention_score = item_skill_attention_score.to_sparse()

            # [item, stu] mm [stu, embed] -> [item, embed]
            stu_embeddings_4_item_fusion = torch.sparse.mm(item_stu_attention_score, self.item_fusion_transformation_layer_4_stu(user_embed))
            # map level attention score
            stu_embeddings_4_item_fusion_attention_input = torch.cat([item_embed, stu_embeddings_4_item_fusion], dim=1) # [item, embed*2]
            stu_embeddings_4_item_fusion_attention_score = self.map_level_item_stu_attention_layer(stu_embeddings_4_item_fusion_attention_input)  # [item, 1]

            # [item, skill] mm [skill, embed] -> [item, embed]
            skill_embeddings_4_item_fusion = torch.sparse.mm(item_skill_attention_score, self.item_fusion_transformation_layer_4_skill(skill_embed))
            # map level attention score
            skill_embeddings_4_item_fusion_attention_input = torch.cat([item_embed, skill_embeddings_4_item_fusion], dim=1) # [item, embed*2]
            skill_embeddings_4_item_fusion_attention_score = self.map_level_item_skill_attention_layer(skill_embeddings_4_item_fusion_attention_input) # [item, 1]


            item_embeddings = item_embed + torch.mul(stu_embeddings_4_item_fusion_attention_score.repeat(1, self.embed_size), stu_embeddings_4_item_fusion) \
                              + torch.mul(skill_embeddings_4_item_fusion_attention_score.repeat(1, self.embed_size), skill_embeddings_4_item_fusion)
            final_item_embeddings = final_item_embeddings + item_embeddings

            '''-------Skill fusion-------'''
            print("-------Knowledge Concept Fusion-------")
            skill_item_attention_score = torch.sparse_coo_tensor(size=(self.skill_num, self.item_num)).to(self.device)
            for j in range(self.skill_num):
                item_index = np.where(q[:, j] == 1)[0]
                one_skill_embed = skill_embed[j].repeat(len(item_index), 1)  # [item, embed]
                one_skill_item_attention_input = torch.cat([self.skill_fusion_transformation_layer_4_item(one_skill_embed),
                                                            self.skill_fusion_transformation_layer_4_item(item_embed[item_index])], dim=1)  #[item, embed*2]
                one_skill_item_attention_score = self.skill_item_attention_layer(one_skill_item_attention_input)  # [item, 1]
                one_skill_item_attention_score = one_skill_item_attention_score.squeeze(-1)  # [item]

                skill_item_attention_score = skill_item_attention_score.to_dense()
                skill_item_attention_score[j, item_index] = one_skill_item_attention_score
                skill_item_attention_score = skill_item_attention_score.to_sparse()

            # [skill, item] mm [item, embed] -> [skill, embed]
            skill_embeddings = skill_embed + torch.sparse.mm(skill_item_attention_score, self.skill_fusion_transformation_layer_4_item(item_embed))  # [skill, embed]
            final_skill_embeddings = final_skill_embeddings + skill_embeddings

        '''Predictions'''
        final_user_embeddings = final_user_embeddings / self.layer_num
        final_item_embeddings = final_item_embeddings / self.layer_num
        final_skill_embeddings = final_skill_embeddings / self.layer_num



        q_vector = q[item]  # [item, skill]
        predict_user_embed = final_user_embeddings[user]  # [bs, embed]
        predict_item_embed = final_item_embeddings[item]  # [bs, embed]

        predict_user_embed = predict_user_embed.unsqueeze(1)  # [bs, 1, embed]
        predict_user_embed = predict_user_embed.repeat(1, self.skill_num, 1)  # [bs, skill, embed]

        predict_item_embed = predict_item_embed.unsqueeze(1).repeat(1, self.skill_num, 1)  # [bs, skill, embed]

        predict_skill_embed = final_skill_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [bs, skill, embed]

        user_proficiency = self.fuse_stu_skill_layer(torch.cat([predict_user_embed, predict_skill_embed], dim=-1))  # [bs, skill, embed]
        user_proficiency = torch.sigmoid(user_proficiency)

        item_difficulty = self.fuse_item_skill_layer(torch.cat([predict_item_embed, predict_skill_embed], dim=-1))  # [bs, skill, embed]
        item_difficulty = torch.sigmoid(item_difficulty)

        prediction = self.performance_prediction_layer((user_proficiency - item_difficulty))  # [bs, skill, 1]
        prediction = torch.sigmoid(prediction.squeeze(-1))  # [bs, skill]

        # [bs, skill] mm [skill, item] -> [bs, item]
        prediction = torch.mm(prediction, torch.FloatTensor(q_vector.T))  # [bs, item]
        prediction = torch.mean(prediction, dim=1)  # [bs]

        def get_cdm_skill(user_embeddings, skill_embeddings):
            user_embeddings = user_embeddings.repeat(1, self.skill_num).view(self.user_num * self.skill_num, -1)  # [stu * skill, embed]
            skill_embeddings = skill_embeddings.repeat(self.user_num, 1)  # [stu * skill, embed]

            user_proficiency = self.fuse_stu_skill_layer(torch.cat([user_embeddings, skill_embeddings], dim=1))
            user_proficiency = torch.sigmoid(user_proficiency) # [stu * skill, embed]

            user_mastery = self.performance_prediction_layer(user_proficiency)
            user_mastery = torch.sigmoid(user_mastery)  # [stu * skill, 1]

            user_mastery = user_mastery.view(self.user_num, self.skill_num)

            return user_mastery

        user_mastery = get_cdm_skill(final_user_embeddings, final_skill_embeddings)

        return prediction, user_mastery





