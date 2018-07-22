# -*- coding: utf-8 -*-
# @Author: yuxiangyu
# @Date:   2018-02-05 13:57:15
# @Last Modified by:   yuxiangyu
# @Last Modified time: 2018-02-05 16:21:58

if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    import pandas as pd
    sys.path.append('/Users/yuxiangyu/Documents/毕业论文-20171126/毕业论文code/')
    from DataAnalysis import DataAnalysis
    from sklearn import neighbors as knn

    # 基本处理
    stat = DataAnalysis()
    now_time = stat.get_time()

    save_columns_name_list = ['k_value', 'KNN', 'delKNN', 'del_num', 'pre_del_num','del_attr', 'data']
    last_result_df = pd.DataFrame(columns=save_columns_name_list)
    # 获取数据集路径
    train_data_path_list = stat.datasets
    for ix, tmp_data_path in enumerate(train_data_path_list):
        # 获取数据集名称 存储结果文件名
        file_name = tmp_data_path.split('/')[-1].split('.')[0]
        # 读取数据
        data_df = stat.open_csv(tmp_data_path)

        # 行列数量
        row_len, col_len = data_df.shape
        # 生成相等权重的数组
        orginal_weight_array = np.array(list([1]* (col_len - 1)))
        orginal_weight_array = orginal_weight_array / sum(orginal_weight_array) 
        # 获取训练集和测试集
        train_data, test_data = stat.get_train_test(data_df)

        # 传统 KNN 算法
        train_array = train_data.drop('label', axis=1)
        test_array = test_data.drop('label', axis=1)
        for k_value in range(1, 21):
            clf = knn.KNeighborsClassifier(n_neighbors = k_value, weights = 'uniform')
            clf.fit(train_data.drop('label', axis = 1), train_data['label'])
            # 预测结果
            predict_label_array = clf.predict(test_array)
            classic_knn_result = stat.get_accuracy_score(predict_label_array, test_data['label'])
            # print('k值:{kvalue}, 分类准确率:{re:.2f} %'.format(kvalue = k_value,re = classic_knn_result * 100)) 
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = k_value), 'k_value'] = k_value
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = k_value), 'KNN'] = round(classic_knn_result * 100, 2)
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = k_value), 'data'] = file_name


        # 中心矩阵
        center_result_df = pd.DataFrame(columns=train_data.columns)
        iter_center_result_df = pd.DataFrame(columns=train_data.columns)
        for class_name, values in train_data.groupby('label'):
            tmp_class_name_list = values.columns
            columns_name_list = [i for i in tmp_class_name_list if 'label' not in i]
            tmp_center_mean_list = list(values[columns_name_list].mean())
            tmp_center_mean_list.append(class_name)
            center_result_df.loc[class_name, :] = tmp_center_mean_list
            tmp_result,iter_n = stat.get_weights_iteration(values[columns_name_list], orginal_weight_array, error_param = 1e-4, max_iter= 10000, alpha=0.05)
            tmp_result.append(class_name)
            iter_center_result_df.loc['{name}:{iter}'.format(name = class_name, iter = iter_n), :] = tmp_result
        iter_without_label = iter_center_result_df[list(iter_center_result_df.columns)[:-1]]
        tmp_iter_std = iter_without_label.std()
        tmp_iter_std.sort_values(inplace= True)
        # center_result_df = center_result_df[list(center_result_df.columns)[:-1]]
        # center_result_df_std = center_result_df.std()
        # center_result_df_std.sort_values(inplace= True)
        
        # 均值
        iter_mean_value = iter_without_label.mean()
        iter_mean_value.sort_values(ascending=False, inplace=True)
        # 贡献较低的数据
        iter_left_series = iter_mean_value.loc[iter_mean_value.cumsum() >= 0.98]
        # 欲删除个数
        prepare_del_num = len(iter_left_series)
        # 求交集
        del_col = list(tmp_iter_std.iloc[:prepare_del_num].index & iter_left_series.index)
        
        # 删除个数
        del_len = len(del_col)
        # del_col = ['col_4','col_6', 'col_7'] # wine
        # del_col = list(tmp_iter_std.iloc[:2].index)
        del_train_data = train_data.drop(del_col, axis=1)
        del_test_data = test_data.drop(del_col, axis=1)
        for del_k_value in range(1, 21):
            del_clf = knn.KNeighborsClassifier(n_neighbors = del_k_value, weights = 'uniform')
            del_clf.fit(del_train_data.drop('label', axis = 1), del_train_data['label'])
            # 预测结果
            del_predict_label_array = del_clf.predict(del_test_data.drop('label', axis=1))
            del_classic_knn_result = stat.get_accuracy_score(del_predict_label_array, del_test_data['label'])
            # print('del_k 值:{kvalue}, 分类准确率:{re:.2f} %'.format(kvalue = del_k_value,re = del_classic_knn_result * 100)) 
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = del_k_value), 'delKNN'] = round(del_classic_knn_result * 100,2)
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = del_k_value), 'del_attr'] = del_col
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = del_k_value), 'del_num'] = del_len
            last_result_df.loc['{data}:{k}'.format(data = file_name, k = del_k_value), 'pre_del_num'] = prepare_del_num


    last_result_df.to_excel('result_all_new.xls')