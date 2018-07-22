# -*- coding: utf-8 -*-
# @Author: yuxiangyu
# @Date:   2017-10-16 10:26:24
# @Last Modified by:   yuxiangyu
# @Last Modified time: 2017-10-23 18:34:41

"""Aw = CI + ß"""

# 构造列名
def get_col_name_func(columns_number):
    """
    输出包含 label 的列名
    """
    col_name_list = ['col-{num}'.format(num = i) for i in range(1, columns_number)]
    col_name_list.extend(['label'])

    return col_name_list


# 定义一个距离函数
def get_weights_dis(trains_datasets, test_values, trains_means):
    tmp = ((trains_datasets * test_values).sum(axis = 1) - trains_means) ** 2
    tmp_index_min = tmp.idxmin()

    return tmp_index_min


# 蒙特卡罗模拟 并计算出 各类的中心点
def set_mtkl_func(train_datasets_df, random_num  = 500, labelname = 'label'):
    """寻找训练集中对应的权重向量"""
    col_name_list = list([i for i in train_datasets_df.columns if 'label' not in i])
    col_name_list.extend(['mean', 'std', 'label'])
    # 构造新的 df 框架
    result_df = pd.DataFrame(columns = col_name_list)
    # 获取训练集维度
    row_tmp, col_tmp = train_datasets_df.shape
    class_center_vector = {}
    # train_datasets_df = train_datasets_df.drop('')
    for class_name, ix in train_datasets_df.groupby(labelname):
        ix = ix.drop(labelname, axis = 1)
        # 添加归一化
        # ix /= ix.sum(axis=1)
        # 标准化 正态 同类中的标准化 不会区分性质 效果不好
        # ix = ix.apply(lambda x: (x - np.mean(x)) / np.std(x))

        # 求均值 
        class_center_vector[class_name] = ix.mean()

        for ii in range(random_num):
            weights = np.random.random(col_tmp - 1)
            weights /= weights.sum()
            b_vector = np.dot(ix, weights.T)
            b_mean = float(b_vector.mean())
            b_std = float(b_vector.std())
            # 存储
            weights_list = list(weights)
            weights_list.extend([b_mean, b_std, class_name])
            result_df.ix['{name}:{number}'.format(name = class_name, number = ii), :] = weights_list

    class_center_vector_df = pd.DataFrame(class_center_vector)
    class_center_vector_df = class_center_vector_df.T # 转置

    return result_df, class_center_vector_df


# 每类提权前 m 个作为新的训练集
def get_new_train_func(weights_mean_std_df, numbers = 5):
    label_weights_number_list = []
    for w_name, w_values in weights_mean_std_df.groupby('label'):
        # 排序
        tmp_label_df = w_values.sort_values('std')[: numbers]
        label_weights_number_list.append(tmp_label_df)

    # 合并
    tmp_result = pd.concat(label_weights_number_list)
    # 重置
    # tmp_result.reset_index().drop('index',axis = 1)

    return tmp_result


# 自定义 KNN 算法
def center_weights_func(trains_df, test_without_label_seris, center_vector_df ,k_values = 1, alpha = 0.5):
    trains_sets_weights = trains_df[[i for i in trains_df.columns if 'col' in i]]
    trains_label = trains_df['label']
    trains_mean_value = trains_df['mean']

    # 计算距离
    tmp_dis = (abs((test_without_label_seris * trains_sets_weights).sum(axis = 1) - trains_mean_value)) ** 2
    tmp_dis = tmp_dis.apply(lambda x: np.sqrt(x))
    # # 排序 并获取前 k 个值
    # tmp_dis_sort_k = tmp_dis.sort_values()[ : k_values]
    # # 获取分类对应标签
    # result_label = list(trains_df.loc[tmp_dis_sort_k.index].label)
    # class_label_number_dict = {result_label.count(keys): keys for keys in list(set(trains_df.label))}

    # # 输出个数最多的一组
    # class_label_result = class_label_number_dict[max(class_label_number_dict.keys())]

    dict_result = {keys.split(':')[0]:value for keys, value in tmp_dis.to_dict().items()}
    weights_series = pd.Series(dict_result)

    """添加算法, 中心距来衡量"""
    test_center_dis = test_without_label_seris - center_vector_df
    test_center_dis = test_center_dis.apply(lambda x: x * x).sum(axis=1) # 求欧式距离
    test_center_dis = test_center_dis.apply(lambda x: np.sqrt(x))

    # 合并两个结果
    weights_series.index = test_center_dis.index
    weigt_tmp_df = weights_series.to_frame(name = 'weights_dis')
    center_tmp_df = test_center_dis.to_frame(name = 'center_dis')

    """对中心距进行归一化计算, 发现效果更佳: 对部分数据效果显著"""
    center_tmp_df = center_tmp_df / center_tmp_df.sum()

    # 可以进行改动的地方 现在用的算术平均 加权线性平均
    result = pd.concat([weigt_tmp_df * (1 - alpha), center_tmp_df * (alpha)], axis=1).sum(axis = 1).idxmin()

    return result


# 要标准化 对训练集
def get_norm_data(trains_datasets_df, method = 'norm'):
    trains_datasets_df_tmp = trains_datasets_df.copy()
    if method is 'norm':
        data_norm_tmp = train_df.drop('label', axis= 1).apply(lambda x: (x - np.mean(x)) / np.std(x))
        data_norm_tmp.loc[:,'label'] = trains_datasets_df.label
        print('执行标准化')
        return data_norm_tmp
    elif method is 'mult':
        data_norm_tmp = train_df.drop('label', axis= 1).apply(lambda x: x**2)
        data_norm_tmp.loc[:,'label'] = trains_datasets_df.label

        return data_norm_tmp


def depuration_alg(train_feature_df, train_label_series, k_value = 3):
    """
    输入:
        train_feature_df df 训练集
        train_label_series label 序列
        train_feature_df.index == train_label_series.index --> True
        method 两个参数:
            'relabel' 对原训练集进行重新定义 label, 从而构造新训练集
            'remove' 移除不满足条件的原训练集点, 从而构造新训练集 (数量会减少)
    输出:
        (1) 新训练集
        (2) 新训练集相比于原训练集, 减少训练集数量
    """
    new_train_feature = train_feature_df.copy()
    # 获取列名 list
    columns_name_list = list(new_train_feature.columns)
    columns_name_list.extend(['label'])
    new_train_df = pd.DataFrame(columns = columns_name_list)

    train_label_series = train_label_series#.astype('str')
    for idx, rows in new_train_feature.iterrows():
        new_train_feature_drop_rows = new_train_feature.drop(idx) # 除 idx 其他的内容
        dis_ = rows - new_train_feature_drop_rows # 离差
        dis_result_sort = dis_.apply(lambda x: x ** 2).sum(axis = 1).sort_values() # 欧式距离
        bool_result = train_label_series[dis_result_sort[ : k_value].index].values == train_label_series[idx]
        # 统计数量
        rows_label_result = list(bool_result).count(True)
        if (rows_label_result >= (k_value + 1)/2) and (rows_label_result <= k_value):
            feature_value_list = list(rows)
            feature_value_list.extend([train_label_series[idx]])
            new_train_df.loc[idx, :] = feature_value_list
    # 度量数据数量
    train_feature_num = len(train_feature_df)
    new_train_df_num = len(new_train_df)
    reduce_num = train_feature_num - new_train_df_num

    return new_train_df, reduce_num


if __name__ == '__main__':
    import pandas as pd
    import sys
    import os
    import numpy as np
    import scipy as sp

    # 导入机器学习库
    from sklearn.cross_validation import train_test_split
    from sklearn import neighbors as knn
    from sklearn.metrics import accuracy_score
    import datetime

    # 文件路径
    data_path1 = 'iris.csv'
    data_path2 = 'seeds_dataset.txt'
    data_path3 = 'new_breast_cancer_data.txt'
    data_path4 = 'new_win_data.txt'
    data_path5 = 'new_glass.txt'
    data_path6 = 'new-segmentation.txt'
    data_path7 = 'new-winequality-white.txt'
    data_path8 = 'Website Phishing.txt'
    data_path9 = 'new-balance-scale.txt'
    
    file_path_list = [data_path1, data_path2, data_path3, data_path4, data_path5, data_path6, data_path7, data_path8, data_path9]
    
    arg_list = np.linspace(0,1,11)
    
    result_df = pd.DataFrame(columns = ['KNN-mean', 'My-mean','KNN-std','My-std'])
    
    
    for data_path in file_path_list[6:]:
        # 文件名
        file_name = data_path.split('/')[-1].split('.')[0]

        # 读取数据 包含 label
        data = pd.read_csv(data_path, header=None)
        # 行数和列数
        row_n, col_n = data.shape
        col_name_label_list = get_col_name_func(col_n)
        # 数据列名重命名
        data.columns = col_name_label_list
        # 训练集 测试集
        data_copy = data.copy()
        data_copy = data_copy.dropna()
        for alpha_value in arg_list:
            alpha_v = alpha_value
            # 根据相关性剔除 列
            # data_copy  = data_copy.drop(['col-5'], axis = 1)

            # 循环多次计算
            classic_knn = []
            my_knn = []

            # 存储结果
            # result_df = pd.DataFrame(columns = ['KNN', 'My'])

            for ii in range(100):

                train_df, test_df = train_test_split(data_copy, test_size = 0.35)

                # 传统 KNN 算法
                clf = knn.KNeighborsClassifier(n_neighbors = 1, weights = 'uniform')
                clf.fit(train_df.drop('label', axis = 1), train_df['label'])

                # 对待测样本进行预测
                predict_label_array = clf.predict(test_df.drop('label', axis = 1))
                # 准确率
                classic_knn_result = accuracy_score(predict_label_array, test_df['label'])


                # 获取新训练集
                # 1. 数据标准化
                # train_df = get_norm_data(train_df, method = 'mult')

                # 2. 剪枝训练集
                # train_df, reduce_num = depuration_alg(train_df.drop('label', axis = 1), train_df.label)

                # 3. 删除列 效果不明显for glass
                # train_df_copy = train_df.copy()
                # train_df_copy = train_df_copy.drop('col-5', axis=1)

                # test_df_copy = test_df.copy()
                # test_df_copy = test_df_copy.drop('col-5', axis=1)

                change_weights_df, center_vector = set_mtkl_func(train_df, random_num = 300)
                new_train_df = get_new_train_func(change_weights_df, numbers = 1)
                # trains_sets_weights = new_train_df[[i for i in new_train_df.columns if 'col' in i]]

                # 存储计算 label 结果
                cal_result_list = []

                # 遍历测试集
                test_df_without_label = test_df.drop('label', axis = 1)
                # 添加归一化
                # test_df_without_label /= test_df_without_label.sum(axis = 1)

                for ix, cont in test_df_without_label.iterrows():
                    result = center_weights_func(new_train_df, cont, center_vector, alpha = alpha_v)
                    # print(result, test_df.loc[ix].label)
                    cal_result_list.append(result)

                # 计算准确率
                my_knn_result = accuracy_score(cal_result_list, test_df['label'])
                combine_list = {'knn': predict_label_array, 'my': cal_result_list, 'test': list(test_df['label'])}
                # result_df.loc[ii, :] = [classic_knn_result, my_knn_result]
                # print(pd.DataFrame(combine_list))
                print('FileName:', file_name,'agrs:',alpha_v,'process --->', ii)
                classic_knn.append(classic_knn_result)
                my_knn.append(my_knn_result)

            result_df.loc[alpha_v, :] = [np.mean(classic_knn), np.mean(my_knn), np.std(classic_knn), np.std(my_knn)]
            # result_df.loc['std', :] = [np.std(classic_knn), np.std(my_knn)]

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_df.to_excel('result-{fname}-{arg}-{time}.xls'.format(fname = file_name, arg = alpha_v,time = now))

        # print('最终结果')
        # print('训练集数量: {rnum}, 测试集数量: {tnum}, 总量: {anum}'.format(rnum = len(train_df) , tnum = len(test_df), anum = len(data_copy)))
        # print('传统算法-准确率: {knn}'.format(knn = np.mean(classic_knn)))
        # print('本文算法-准确率: {vknn}'.format(vknn = np.mean(my_knn)))
        # # print(len(train_df), len(test_df),len(data_copy),'\n',np.mean(classic_knn), np.mean(my_knn))
        # print('传统算法标准差:', np.std(classic_knn), '本文算法标准差:', np.std(my_knn))
        # print(new_train_df)









