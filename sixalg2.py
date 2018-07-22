"""
程序算法, 对 KNN 算法进行简要的更改.
1. 添加了各个类的中心向量
2. 对测试样本的邻近训练样本进行更全面的考察


算法1 都是基于少数服从多数的原则来进行实验.
即: 
a) 待测样本与不同类中心的距离, 就近选择;
b) 待测样本邻近训练集样本与不同类中心的距离, 少数服从多数;

如果 a = b: 替换 kNN 分类结果, 否则用 KNN 分类
"""

# 构造列名函数
def get_col_name_func(columns_number):
    """
    输出包含 label 的列名
    """
    col_name_list = ['col-{num}'.format(num = i) for i in range(1, columns_number)]
    col_name_list.extend(['label'])

    return col_name_list

# 定义处理字典函数 统计类对应的数量
def handle_dict_func(data_dict):
    data_length = len(data_dict)
    data_set_value_list = list(set(data_dict.values()))
    data_values_list = list(data_dict.values())
    result_dict = {}
    for ix in data_set_value_list:
        result_dict[ix] = data_values_list.count(ix)
    result_df = pd.Series(result_dict)
    
    return result_df.idxmax()  

# 定义 获得不同类中心坐标向量
def get_class_center_vector(train_df):
    result_dict = {}
    for class_label, context in train_df.groupby('label'):
        context_drop_label = context.drop('label', axis = 1)
        temp_center_mean = context_drop_label.mean()
        result_dict[class_label] = temp_center_mean
    #合并数据 并转置
    trains_center_df = pd.DataFrame(result_dict)
    trains_center_df = trains_center_df.T
    
    return trains_center_df

# 训练集与类中心坐标的最小距离
def trains_cond_class_vector_dis(trains_center_df, trains_conditon_data_df):
    # 训练集与类中心坐标的最小距离
    condition_data_center_dis_dict = {}
    for ix , context in trains_conditon_data_df.iterrows():
        tmp_dis = context - trains_center_df
        tmp_dis = tmp_dis.apply(lambda x: x*x).sum(axis = 1)
        condition_data_center_dis_dict[ix] = tmp_dis.idxmin()

    return condition_data_center_dis_dict

# 定义测试样本与类中心距离的函数
def test_center_dis(test_without_label, trains_center_df):
    tmp_class_name = (test_without_label - trains_center_df).apply(lambda x:x*x).sum(axis = 1).idxmin()
    
    return tmp_class_name


# 定义 my KNN alg
def my_knn_alg(test_without_label, train_df, trains_center_df, k_value = 3):
    # 训练集数据 测试数据
    test_without_label = test_without_label
    train_without_label = train_df.drop('label', axis=1)
    train_label = train_df.label
    # 距离 测试与训练
    train_test_dis = test_without_label - train_without_label
    # 欧式距离
    train_test_dis = train_test_dis.apply(lambda x: x * x).sum(axis=1)
    temp_condition_index_list = train_test_dis.sort_values()[:k_value].index
    
    # 训练集中对应的数据
    trains_conditon_data_df = train_without_label.loc[temp_condition_index_list]
    # knn 结果
    knn_result = handle_dict_func(train_df.loc[temp_condition_index_list].label.to_dict())
    
    condition_data_center_dis_dict = trains_cond_class_vector_dis(trains_center_df, trains_conditon_data_df)
    # 测试样本附近的训练样本与类中心的距离 最小距离
    test_nearby_trains_center_dis = handle_dict_func(condition_data_center_dis_dict)
    # 测试样本与类中心的距离
    test_center_dis_result = test_center_dis(test_without_label, trains_center_df)
    
    if test_center_dis_result == test_nearby_trains_center_dis:
        return test_center_dis_result
    
    return knn_result
    
    
    
if __name__ == '__main__':

    import pandas as pd
    import sys
    import os
    import numpy as np
    import scipy as sp
    import datetime
    # 导入机器学习库
    from sklearn.cross_validation import train_test_split
    from sklearn import neighbors as knn
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # 文件路径
    data_path1 = 'C:\work\jakcyu\Six\datasets\Iris\iris.csv'
    data_path2 = 'C:\work\jakcyu\Six\datasets\seeds\seeds_dataset.txt'
    data_path3 = 'C:\work\jakcyu\Six\datasets\breast_cancer\new_data.txt'
    data_path4 = 'C:\work\jakcyu\Six\datasets\Wine\new_win_data.txt'
    data_path5 = 'C:\work\jakcyu\Six\datasets\Glass\new_glass.txt'
    data_path6 = 'C:\work\jakcyu\Six\datasets\Winequality\new-winequality-red.txt'
    data_path7 = 'C:\work\jakcyu\Six\datasets\Winequality\new-winequality-white.txt'
    data_path8 = 'C:\work\jakcyu\Six\datasets\segmentation\new-segmentation.txt'
    data_path9 = 'C:\work\jakcyu\Six\datasets\balance_scale\new-balance-scale.txt'
    data_path = [data_path1, data_path2, data_path3, data_path4, data_path5, data_path6, data_path7, data_path8, data_path9]
    
    total_result = pd.DataFrame(columns=['knn-mean', 'myknn-mean', 'knn-std', 'myknn-std'])
    
    for datapath in data_path[:1]:
        # 读取数据 包含 label
        data = pd.read_csv(datapath, header=None)
        # 行数和列数
        row_n, col_n = data.shape
        col_name_label_list = get_col_name_func(col_n)
        # 数据列名重命名
        data.columns = col_name_label_list
        # 复制数据集 并 进行随机获取训练集和测试集
        data_copy = data.copy()
        # 数据集名称
        file_name = datapath.split('/')[-1].split('.')[0]
        
        # 设定参数
        # k_value = 7
        test_times = 3
        
        # 遍历 K 值
        for kvalue in range(1, 4):
            #total_result = pd.DataFrame(columns=['KNN-mean', 'MyKNN-mean', 'knn-std', 'myknn-std'])
            classic_knn_accuracy = []
            my_knn_accuracy = []
            for ix in range(test_times):
                # 训练集 测试集
                train_df, test_df = train_test_split(data_copy, test_size = 0.35)
                trains_center_df = get_class_center_vector(train_df)
                # 训练集数据 测试数据
                test_without_label = test_df.drop('label', axis=1)
                test_label = test_df.label
                train_without_label = train_df.drop('label', axis=1)
                train_label = train_df.label
                
                my_knn_alg_result_list = []
                for ii, context in test_without_label.iterrows():
                    tmp_result = my_knn_alg(context, train_df, trains_center_df, k_value=kvalue)
                    my_knn_alg_result_list.append(tmp_result)
                
                # 传统算法
                clf = knn.KNeighborsClassifier(n_neighbors = kvalue, weights = 'uniform')
                clf.fit(train_without_label, train_label)
                predict_label_array = clf.predict(test_without_label)
                predict_label = list(predict_label_array)
                
                classic_knn = accuracy_score(predict_label, test_label)
                my_knn = accuracy_score(my_knn_alg_result_list, test_label)
                # total_result.loc[ix,:] = [classic_knn, my_knn]
                classic_knn_accuracy.append(classic_knn)
                my_knn_accuracy.append(my_knn)
                
                # print('residue: {ixx},样本: {num},传统 KNN 结果:{knn}, 实验 KNN 结果:{mknn}'.format(knn = classic_knn, mknn = my_knn, num = ii ,ixx =test_times- ix ))
            # 计算均值和标准差
            orig_knn_mean = np.mean(classic_knn_accuracy)
            orig_knn_std = np.std(classic_knn_accuracy)
            my_knn_mean = np.mean(my_knn_accuracy)
            my_knn_std = np.std(my_knn_accuracy)
            
            temp_output_result = [orig_knn_mean, my_knn_mean, orig_knn_std, my_knn_std]
            # print(temp_output_result)
            # total_result = total_result.copy()
            total_result.loc['{filen}:{k}'.format(filen = file_name, k = kvalue),:]  = temp_output_result
            # total_result.loc['std',:]  = total_result.std()
    total_result.to_excel('result_kvalue.xls')