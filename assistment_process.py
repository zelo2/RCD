import pandas as pd
import numpy as np
import random
from progressbar import *


''' Dataset Description:
    order_id:时间序列id，These id's are chronological, and refer to the id of the original problem log.
    assignment_id:每个Assignment对应一个独立的老师/班级
    user_id:学生id
    problem_id:问题id
    original:Main problem or Scaffolding problem
    correct:第一次作答结果-正确/不正确/寻求帮助
    attempt_count:学生尝试的次数
    ms_first_response:学生第一次反应时间
    tutor_mode:tutor or test
    answer_type: choose_1 or algebra or fill_in or open_response
    sequence_id:问题集的id
    student_class_id:班级id
    position:课堂作业页面上的作业位置
    problem_set_types:Linear or Random or Mastery
    base_sequence_id:如果序列已经被复制，则指向原始副本
    skill_id:与问题相关的技能ID。在这个技能构建器数据集中，记录将被复制，以便每个记录具有一个技能
    skill_name:知识点名字
    teacher_id:教师id
    school_id:学习id
    hint_count:学生尝试的次数
    hint_total:关于这个问题的可能提示的数量
    overlap_time:毫秒
    template_id:The template ID of the ASSISTment. ASSISTments with the same template ID have similar questions.
    answer_id:The answer ID for multi-choice questions.
    answer_text:The answer text for fill-in questions.
    first_action:The type of first action: attempt or ask for a hint.
    bottom_hint:Whether or not the student asks for all hints.
    opportunity:学生需要在该知识点上练习的机会数
    opportunity original:The number of opportunities the student has to practice on this skill counting only original problems.
'''


def as19_preprocess(data):
    # data = pd.read_csv('2009_skill_builder_data_corrected/skill_builder_data_corrected1.csv', encoding="utf-8")
    # data = pd.read_csv('skill_builder_data_corrected1.csv', encoding="utf-8")
    data.sort_values(axis=0, by='order_id', ascending=True)
    index_column = ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id']
    score_record = data[index_column]


    score_record_np = score_record.values
    score_record_np = score_record_np[:338001]  # 338001之后的skill没有id
    stu_id = np.unique(score_record_np[:, 1])  # 从小到大排序后的学生id，且除去了重复id
    item_id = np.unique(score_record_np[:, 2])  # 试题id
    skill_id = np.unique(score_record_np[:, 4])  # 知识点id

    score_record_np = score_record_np[score_record_np[:, 0].argsort()]  # numpy矩阵根据order_id进行整体排序
    score_record_np = score_record_np[:, 1:]
    # print(np.where((np.isnan(score_record_np[:, 3]) * 1 == 1))[0])



    stu_num = len(stu_id)
    item_num = len(item_id)
    skill_num = len(skill_id)

    for i in range(len(score_record_np)):
        index_stu_id = np.where(stu_id == score_record_np[i][0])[0]  # 遍历每一行，找到该学生id大小排第几
        score_record_np[i][0] = index_stu_id[0]
        index_item_id = np.where(item_id == score_record_np[i][1])[0]
        score_record_np[i][1] = index_item_id[0]
        index_skill_id = np.where(skill_id == score_record_np[i][3])[0]
        score_record_np[i][3] = index_skill_id[0]

    score = np.zeros([stu_num, item_num])
    score[:, :] = np.nan
    q = np.zeros([item_num, skill_num])
    for i in range(len(score_record_np)):
        temp_stu_id = int(score_record_np[i][0])
        temp_item_id = int(score_record_np[i][1])
        temp_score = int(score_record_np[i][2])
        temp_skill_id = int(score_record_np[i][3])
        if np.isnan(score[temp_stu_id][temp_item_id]):
            score[temp_stu_id][temp_item_id] = temp_score
        q[temp_item_id][temp_skill_id] = 1
    indicator = np.zeros(score.shape)
    indicator_index = np.isnan(score) * 1
    indicator_index = np.where(indicator_index == 0)
    indicator_index = np.array(indicator_index)
    indicator_index = indicator_index.T
    for i in indicator_index:
        x = i[0]
        y = i[1]
        indicator[x][y] = 1

    # np.savetxt("2009_skill_builder_data_corrected/as09_indicator.txt", indicator, fmt="%.f")
    # np.savetxt("2009_skill_builder_data_corrected/score.txt", score, fmt="%.f")
    # np.savetxt("2009_skill_builder_data_corrected/q.txt", q, fmt="%.f")
    return score, q, indicator

def as19_data_split(score, q, percent=0.8):

    stu_num = len(score)
    item_num = len(score[0])
    skill_num = len(q[0])

    score_nan_mark = 1 - np.isnan(score) * 1
    stu_response_length = np.sum(score_nan_mark, axis=1)  # sum each row
    delete_list = np.where(stu_response_length < 15)[0]
    raw_data = np.delete(score, delete_list, 0)  # delete based on row
    # raw_data 删除答题数量少于15的学生后的得分矩阵
    # 清理raw_data中没有答题记录的试题，并删除对应的Q矩阵向量
    score_nan_mark = 1 - np.isnan(raw_data) * 1
    item_response_length = np.sum(score_nan_mark, axis=0)  # sum each column
    delete_list = np.where(item_response_length <= 0)[0]
    clean_data = np.delete(raw_data, delete_list, 1)  # delete based on column
    clean_q = np.delete(q, delete_list, 0)  # delete based on row
    # print(clean_data.shape, clean_q.shape)



    indicator = np.zeros(clean_data.shape)
    indicator[:, :] = np.nan
    train_data = []
    test_data = []
    for i in range(len(clean_data)):
        stu_response_mark = 1 - np.isnan(clean_data[i]) * 1
        stu_response_index = np.where(stu_response_mark == 1)[0]
        if len(stu_response_index) < 15:
            print("Warning! Error Data Insert!")
            print(i)
        train_index = random.sample(range(len(stu_response_index)), int(len(stu_response_index) * percent))
        test_index = np.delete(range(len(stu_response_index)), train_index)
        train_item_index = stu_response_index[train_index]
        test_item_index = stu_response_index[test_index]

        indicator[i][train_item_index] = 1
        indicator[i][test_item_index] = 0

        for j in train_item_index:
            train_data.append(np.array([i, j, clean_data[i, j]]).astype('int64'))
        for j in test_item_index:
            test_data.append(np.array([i, j, clean_data[i, j]]).astype('int64'))

    random.shuffle(train_data)
    random.shuffle(test_data)


    return clean_data, clean_q, indicator, train_data, test_data


def as_data_process(data, percent):
    score, q, indicator = as19_preprocess(data)

    clean_data, clean_q, indicator, train_data, test_data = as19_data_split(score, q, percent=percent)
    return clean_data, clean_q, indicator, train_data, test_data





