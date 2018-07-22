# -*- coding: utf-8 -*-
# @Author: yuxiangyu
# @Date:   2017-10-24 19:47:22
# @Last Modified by:   yuxiangyu
# @Last Modified time: 2018-02-05 13:53:56

"""
本函数库为优化库

本库为数据分析部分, 作为毕业论文 数据集提取, 算法实现等集成库
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn import manifold


class DataAnalysis:
    """docstring for DataAnalysis"""
    def __init__(self):
        super(DataAnalysis, self).__init__()
        # info
        self.name = 'jackyu'
        self.email = 'xiangyuyu819@yeah.net'
        # 数据集
        self.datasets = self.data_sets_path()
        # 存储时间点
        self.nowtime = self.get_time()

    # 当前时间的格式
    def get_time(self, out_style='STR'):
        now_time = datetime.datetime.now()
        if out_style is 'STR':
            return now_time.strftime("%Y%m%d%H%M%S")
        elif out_style is 'YMDHMS':
            return now_time
        elif out_style is 'YMD':
            return datetime.date(now_time.year, now_time.month, now_time.day)
        else:
            return "The args of out_style must be one of in ['STR', 'YMDHMS', 'YMD']!!"

    # 数据集路径
    def data_sets_path(self):
        # 可扩充
        path1 = '/Users/yuxiangyu/Documents/论文材料/实验室/Iris/iris.csv'
        path2 = '/Users/yuxiangyu/Documents/论文材料/实验室/Winequality/new-winequality-white.txt'
        path3 = '/Users/yuxiangyu/Documents/论文材料/实验室/Winequality/new-winequality-red.txt'
        path4 = '/Users/yuxiangyu/Documents/论文材料/实验室/Glass/new1-new-glass.txt'
        path5 = '/Users/yuxiangyu/Documents/论文材料/实验室/seeds/seeds_dataset.txt'
        path6 = '/Users/yuxiangyu/Documents/论文材料/实验室/breast_cancer/new-breast-cancer-wisconsin.txt'
        path7 = '/Users/yuxiangyu/Documents/论文材料/实验室/Wine/new-win.txt'
        path8 = '/Users/yuxiangyu/Documents/论文材料/实验室/balance_scale/new-balance-scale.txt'
        path9 = '/Users/yuxiangyu/Documents/论文材料/实验室/segmentation/new-segmentation.txt'
        path_list = [path1, path2, path3, path4,
                    path5, path6, path7, path8, path9]

        return path_list

    # 数据读取方式
    def open_csv(self, file_path, header=None, seq=None, label=True):
        # 数据读取并重命名
        if header is None:  # 不含表头
            data = pd.read_csv(file_path, header=header, delimiter=seq)
            return self.df_col_rename(data, label=label)
        elif seq is not None:
            data = pd.read_csv(file_path, header=header, seq=seq)
            return self.df_col_rename(data, label=label)
        elif (header is 0):  # 含表头
            return pd.read_csv(file_path, header=header, delimiter=seq)
        else:
            return "The data could not be open by the function!!"

    # df 数据重命名
    def df_col_rename(self, data_df, label=True):
        data_df_dim = data_df.columns.size
        if label is True:
            data_col_name_list = ['col_{num}'.format(
                num=i) for i in range(1, data_df_dim)]
            data_col_name_list.extend(['label'])
            data_df.columns = data_col_name_list
            return data_df
        data_col_name_list = ['col_{num}'.format(
            num=i) for i in range(1, data_df_dim + 1)]
        data_df.columns = data_col_name_list
        return data_df

    # 设置 df 框架
    def get_frames(self, dict_data=None, columns=None, index=None):
        if (dict_data is None) and (index is not None) and (columns is not None):
            return pd.DataFrame(columns=columns, index=index)
        elif dict_data is not None:
            return pd.DataFrame(dict_data)
        elif (index is None) and (dict_data is not None) and (columns is not None):
            return pd.DataFrame(dict_data, columns=columns)
        elif (index is None) and (dict_data is None) and (columns is not None):
            return pd.DataFrame(columns=columns)
        else:
            return "dict_data, colums and index is empty!!!"

    # 输出结果扩充
    def out_put_result(self, list_name=None):
        columns_name_list = ['accuracy', 'k_value', 'cost_time']
        if list_name is None:
            return self.get_frames(columns=columns_name_list)
        columns_name_list.extend(list_name)
        return self.get_frames(columns=columns_name_list)

    # 构造训练集和测试集
    @staticmethod
    def get_train_test(data1, data2=None, test_size=0.35, random_state=10):
        '''
        输入:
            data1 数据集(或单个训练集)
            data2 测试集(或数据集中一部分)
            test_size 默认输出 35%
            random_state 随机状态, 默认 10 
        输出:
            随机输出 训练集 和 测试集
            若是2个数据, 则会分别输出2个数据的 训练集 和 测试集 4个数据
        '''
        from sklearn.cross_validation import train_test_split
        if data2 is None:
            return train_test_split(data1, test_size=test_size, random_state=random_state)
        return train_test_split(data1, data2, test_size=test_size, random_state=random_state)

    # 结果得分 验证正确率
    @staticmethod
    def get_accuracy_score(predict_result, control_result):
        '''
        输入:
            predict_result 预测结果 list , series, array_like 
            control_result 对照数据 list , series, array_like
        输出:
            准确率
        '''
        from sklearn.metrics import accuracy_score

        return accuracy_score(predict_result, control_result)

    # 将数据进行合并
    @staticmethod
    def combine_data_func(data1_df, data2_df, axis=1, fs='outer'):
        '''
        data1_df,2 均为 df 数据格式
        style 按照行列合并, 1 按行合并, 0 按列合并, 默认为1
        fs = inner 求交集, outer 求并集 
        '''
        if fs in ['outer', 'inner']:
            result = pd.concat([data1_df, data2_df], axis=axis, join=fs)
            return result
        else:
            return "The args of fs is ERROR, it's must be one of ['outer','inner']!! "

    # @ jackyu 以上代码优化时间: 2017-10-24 21:21:21

    # 计算 values 闵可夫斯基距离(Minkowski Distance)
    def dis_minkowski_series(self, series_1, series_2, p_value=2):
        """
        输入:
            2个 float(int) value or series or vector
        输出:
            p_value 参数的 闵可夫斯基距离
        """
        two_value_dis = pow(
            sum(pow(abs(series_1 - series_2), p_value)), 1 / p_value)

        return two_value_dis

    # 计算多维 df 闵可夫斯基距离
    # @staticmethod
    def dis_minkowski_df(self, df_1, df_2, p_value=2, method='normal', W=None):
        # df 数据 快速计算 normal 表示正常的闵式距离
        if (method is 'normal') and (W is None):  # sqrt(sum((x - y)^2))
            tmp_dis = (
                df_1 - df_2).apply(lambda x: pow(abs(x), p_value)).sum(axis=1)
            return tmp_dis.apply(lambda x: np.sqrt(x))
        elif (method is 'chebyshev') and (W is None):  # max(|x - y|)
            tmp_dis = (df_1 - df_2).apply(lambda x: abs(x)).max(axis=1)
            return tmp_dis
        elif method is 'minkowski':  # sum(w * |x - y|^p)^(1/p)
            if W is None:
                tmp_dis = pow(
                    (df_1 - df_2).apply(lambda x: pow(abs(x), p_value)).sum(axis=1), 1 / p_value)
                return tmp_dis
            tmp_dis = pow((df_1 - df_2).apply(lambda x: pow(W *
                                                            abs(x), p_value)).sum(axis=1), 1 / p_value)
            return tmp_dis
        elif (method is 'cosine') and (W is None):  # 余弦相似度(仅两个向量做)
            tmp_dis = df_1.dot(df_2) / (pow(sum((df_1 ** 2)),
                                            1 / 2) * pow(sum(df_2 ** 2), 1 / 2))
            return tmp_dis
        else:
            return """
                    The args of method is one of ['normal', 'chebyshev', 'wminkowski', 'cosine'], and 
                     ==============  ====================  ========  ===============================   
                     identifier      class name            args      distance function                
                     --------------  --------------------  --------  -------------------------------  
                     "normal"        EuclideanDistance     -         ``sqrt(sum((x - y)^2))``         
                     "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``                 
                     "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``                 
                     "minkowski"     MinkowskiDistance     p         ``sum(|x - y|^p)^(1/p)``         
                     "wminkowski"    WMinkowskiDistance    p, w      ``sum(w * |x - y|^p)^(1/p)``     
                     ==============  ====================  ========  ===============================  
                    """

    """    KNN 算法     """
    # 自编 KNN 算法

    def method_my_knn_func(self, test_series, train_df, p_value=2, k_value=1, method='normal', weigths=None, out_put='uniform'):

        train_df_without_label = train_df.drop('label', axis=1)
        train_series_label = train_df.label
        if method is 'cosine':
            return "The input data must be vector!!!"
        tmp_result_series = self.dis_minkowski_df(
            test_series, train_df_without_label, p_value=p_value, W=weigths, method=method)
        top_k_series = tmp_result_series.sort_values()[: k_value]

        # 调用函数
        result_df = self.sub_base_dis(top_k_series, train_series_label)
        if (out_put is 'uniform') and (k_value % 2 == 1):  # 少数服从多数的情况 ()
            return result_df.index[0]
        elif (out_put is 'uniform') and (k_value % 2 == 0):  # 若为偶数
            if result_df['count'].iloc[0] / k_value > 0.5:
                return result_df.index[0]
        else:
            print('存在投票数量一致现象!')
            return result_df

    # 定义距离筛选
    def sub_base_dis(self, top_k_index_dis, train_series_label):
        top_k_index_label = train_series_label[top_k_index_dis.index]
        top_k_index_dis.rename('dis', inplace=True)
        # 合并df
        tmp_combine_df = self.combine_data_func(
            top_k_index_dis, top_k_index_label)
        # 创建输出 df 框架
        tmp_result_df = self.get_frames(
            columns=['count', 'dis_sum', 'dis_mean'])
        for name, rows in tmp_combine_df.groupby('label'):
            rows_without_label = rows.drop('label', axis=1)
            rows_without_label_sum = rows_without_label.sum().dis
            tmp_result_df.loc[name, :] = len(
                rows), rows_without_label_sum, rows_without_label_sum / len(rows)

        return tmp_result_df.sort_values('count', ascending=False)

    # 聚类 K-means()
    @staticmethod
    def use_k_means(data_array, test_array=None, cluster_num=3):
        '''
        默认下的设置, 在使用中不再对其进行设置
        init='k-means++', n_init=10, max_iter=300, 
        tol=0.0001, precompute_distances='auto', 
        verbose=0, random_state=None, copy_x=True, 
        n_jobs=1, algorithm='auto
        '''
        # 导入 KMeans 聚类库
        from sklearn.cluster import KMeans
        if len(data_array) >= cluster_num:
            kmeans = KMeans(n_clusters=cluster_num,
                            random_state=10).fit(data_array)
            if test_array is None:
                return kmeans.labels_, kmeans.cluster_centers_
            else:
                return kmeans.labels_, kmeans.cluster_centers_, kmeans.predict(test_array)
        else:
            kmeans = np.array(data_array)
            if test_array is None:
                return 'No_label', kmeans
            else:
                return "Can't to cluster!"

    # 任意两个做比较的函数 慎用 复杂度很高(不适合样本量较大数据)
    def get_comp_func(self, data_df, method=2):
        '''
        data_df 为一个 DataFrame 框架的数据 包含多行
        默认状态下用 欧式距离 method =2
        输出 data_df 格式
        '''
        if len(data_df) == 1:
            return '数据仅有1行, 不满足计算条件'
        temp_list = []
        sum_n = 0
        for ix in data_df.index[:-1]:
            sum_n += 1
            for ii in data_df.index[sum_n:]:
                dis = self.dis_minkowski_series(
                    data_df.loc[ix], data_df.loc[ii], p_value=method)
                temp_list.append([(ix, ii), dis])
        # 存储 df 格式
        result_df = self.get_frames(columns=['from_value', 'to_value', 'dis'])
        for ix, tl in enumerate(temp_list):
            (from_value, to_value), dis = tl
            result_df.loc[ix, ['from_value', 'to_value', 'dis']
                          ] = from_value, to_value, dis

        return result_df

    # 定义个人的 kNN 算法 为研究权重而用
    # @staticmethod
    def my_knn_alg_func(self, train_df_with_label, test_series_without_label, k_value=3, p_value=2):
        '''
        输入: 
            带标签的 train_df_with_label 训练集数据
            无标签的 test_series_without_label 待测数据(或待分类数据)
            k_value 选取 k 个近邻
            p_value 距离, 默认情况下为欧式距离
        输出:
            df 格式的 k 个近邻标签 以及 距离
        '''
        columns_name_list = train_df_with_label.columns
        train_data = train_df_with_label[columns_name_list[:-1]]
        train_label = train_df_with_label[columns_name_list[-1]]
        if train_data.columns.shape == test_series_without_label.shape:
            minkowski_dis_series = self.dis_minkowski_df(
                test_series_without_label, train_data, p_value=p_value)
            dis_sort_series_k = minkowski_dis_series.sort_values()[:k_value]
            # 获取对应的 label
            train_label = train_label[dis_sort_series_k.index].values
            # 压缩 to list
            list_tuple_result = list(
                zip(train_label, dis_sort_series_k.values))
            # 存储 df
            df = pd.DataFrame(columns=['dis', 'label'])
            for ix, cont in enumerate(list_tuple_result):
                df.loc[ix, :] = [cont[-1], cont[0]]
            return df
        else:
            print('test_data shape is: {0}, train_data shape is: {1}, so do not calus!'.format(
                test_series_without_label.shape, train_data.shape))
            return None

    # 定义一个双加权函数
    """文献: a new distance-weighted k-nearest Neighbor classifier"""
    @staticmethod
    def get_dual_weights(x_df_value, style='normal', drop_col='label', right=1):
        '''
        输入: 
            x_df_value 为多行或1行 2 列数据, 列名['dis','label'] 即要求数值型 和 字符 两种格式
        输出:
            同类型的 df
        '''
        if len(x_df_value) is not 1:
            x_df_value_without_label = x_df_value.drop(drop_col, axis=1)
            result_df_left = x_df_value_without_label.apply(
                lambda x: (max(x) - x) / (max(x) - min(x)))
            result_df_right = x_df_value_without_label.apply(
                lambda x: (max(x) + min(x)) / (max(x) + x))
            # 测试指数结果
            # dual_weights = np.exp(result_df_left) * (result_df_right ** right)
            dual_weights = result_df_left * (result_df_right ** right)
            dual_weights[drop_col] = x_df_value[drop_col]
            return dual_weights
        else:
            x_df_value.dis = 1
            return x_df_value

    # 定义 统计类别对应的个数
    @staticmethod
    def get_arg_max_min_func(data_df, sort_arg=True):
        '''
        输入:
            data_df 为两列 df 数据, 第一列为数值, 第二列为 label
            sort_arg 默认为 True , 降序, 输出最大值对应的 label
        输出:
            输出 label
        '''
        arg_max_dict = {}
        columns_list = data_df.columns
        if 'label' in columns_list:
            class_label_list = data_df.label.drop_duplicates()
            # 若只有一个类 label ,直接输出
            if len(class_label_list) == 1:
                return class_label_list[0]
            else:
                for ix in class_label_list:
                    temp_value_with_label = data_df.loc[data_df.label == ix]
                    # 排除 label 列
                    arg_max_dict[ix] = temp_value_with_label[
                        columns_list[0]].sum()
                # 对字典排序
                max_num_label = sorted(arg_max_dict.items(), key=lambda x: x[
                                       1], reverse=sort_arg)
                # 获取最大值的 label
                max_label = max_num_label[0][0]
                return max_label
        else:
            return "input_data does not contain 'label'or other error!"

    # LLE 算法
    @staticmethod
    def method_lle_func(data_df, test_df=None, n_k=3, n_c=2, e_s='dense'):
        """        
        输入:
            data_df 表示所有数据
            n_k 近邻数
            n_c 选择嵌入数量
            e_s 求解方法
        输出:
            已转化数据
            损失函数值
        """
        import sklearn.manifold as sm
        model = sm.LocallyLinearEmbedding(
            n_neighbors=n_k, n_components=n_c, eigen_solver=e_s)
        model.fit(data_df)
        if test_df is None:
            return model
        else:
            return model.fit_transform(test_df)

    # lle 算法 改进版
    @staticmethod
    def improve_lle_alg(train_df, test_df=None, n_k=3, n_c=2, e_s='dense'):
        """
        输入:
            train_df 训练集; test_df 测试集; 均为 feature 数据, 无 label.
            n_k lle 选取近邻数, 对应 n_neighbors;
            n_c 选取嵌入数, 对应 n_components
            e_s 特征值求解算法, eigen_solver 默认为 dense, 必能找到求解方式, 详细请 help
        输出: 
            W 矩阵, 转化矩阵
            求解 W, 线性表示的误差项.
        """
        transform_array, error_result = manifold.locally_linear_embedding(X=train_df, n_neighbors=n_k,
                                                                          n_components=n_c, eigen_solver=e_s)
        # 数据类型转化
        train_data_transform_df = pd.DataFrame(transform_array)  # array to  df
        reduce_dim_matrix = np.matrix(transform_array)
        train_data_matrix = np.matrix(train_df.astype('float'))  # 是否存在逆??
        train_data_matrix_inv = train_data_matrix.getI()  # 求逆(不是方阵)
        # 计算出 W 矩阵
        w_transform_matrix = train_data_matrix_inv * reduce_dim_matrix
        # 判断是否输入测试集
        if test_df is None:  # 输出: 局部线性嵌入转化后的训练集,W 矩阵, 误差项
            return train_data_transform_df, w_transform_matrix, error_result
        test_data_transform_matrix = np.matrix(test_df) * w_transform_matrix
        test_data_transform_df = pd.DataFrame(test_data_transform_matrix)
        # 输出4个指标, 转化后的train, test, W 矩阵, 以及 误差项
        return train_data_transform_df, test_data_transform_df, w_transform_matrix, error_result

    # Depuration 算法
    """文献: Editing training data for KNN classifiers with neural network ensemble"""
    # @staticmethod

    def depuration_alg(self, train_feature_df, train_label_series, k_value=3, p_value=2):
        """
        输入:
            train_feature_df df 训练集
            train_label_series label 序列
            train_feature_df.index == train_label_series.index --> True
        输出:
            新训练集  新训练集相比于原训练集, 减少训练集数量
        """
        new_train_feature = train_feature_df.copy()
        train_label_series = train_label_series.astype('str')
        # 获取列名 list
        columns_name_list = list(new_train_feature.columns)
        columns_name_list.extend(['label'])
        # 定义存储 list
        temp_list = []
        index_list = []
        for idx, rows in new_train_feature.iterrows():
            feature_value_list = None
            new_train_feature_drop_rows = new_train_feature.drop(
                idx)  # 除 idx 其他的内容
            dis_ = self.dis_minkowski_df(
                rows, new_train_feature_drop_rows, p_value=p_value)
            dis_result_sort = dis_.sort_values()  # 排序
            bool_result = train_label_series[dis_result_sort[
                : k_value].index].values == train_label_series[idx]
            # 统计数量
            rows_label_result = list(bool_result).count(True)
            if (rows_label_result >= (k_value + 1) / 2) and (rows_label_result <= k_value):
                feature_value_list = list(rows)  # 特征向量
                feature_value_list.extend([train_label_series[idx]])  # 标签
                temp_list.append(feature_value_list)
                index_list.append(idx)
        # 转化 df
        new_train_df = pd.DataFrame(
            data=temp_list, index=index_list, columns=columns_name_list)
        # 度量数据数量
        train_feature_num = len(new_train_feature)
        new_train_df_num = len(new_train_df)
        reduce_num = train_feature_num - new_train_df_num

        return new_train_df, reduce_num

    # 最后编辑时间 20170819

    # 要标准化 对训练集
    @staticmethod
    def get_norm_data(trains_datasets_df, method='norm'):
        """    
        标准化数据, norm 标准化得分; mult 使得>1的放大, <1 的缩小
        """
        trains_datasets_df_tmp = trains_datasets_df.copy()
        if method is 'norm':
            data_norm_tmp = train_df.drop('label', axis=1).apply(
                lambda x: (x - np.mean(x)) / np.std(x))
            data_norm_tmp.loc[:, 'label'] = trains_datasets_df.label
            print('执行标准化')
            return data_norm_tmp
        elif method is 'mult':
            data_norm_tmp = train_df.drop(
                'label', axis=1).apply(lambda x: x**2)
            data_norm_tmp.loc[:, 'label'] = trains_datasets_df.label

            return data_norm_tmp

    # 通过迭代法来求解权重
    @staticmethod
    def get_weights_iteration(rows_without_label, original_weights, error_param=1e-4, max_iter=300, alpha=0.05):
        columns_name_list = rows_without_label.columns
        # 初始状态
        error_std = max(rows_without_label.std()) - \
            min(rows_without_label.std())
        # 初始化迭代次数
        iter_count = 0
        while (error_std >= error_param) and (iter_count <= max_iter):
            temp_result = rows_without_label * original_weights
            # 计算各特征标准差
            std_result = temp_result.std()
            index_max = std_result.idxmax()
            index_id = list(columns_name_list).index(index_max)
            # original_weights[index_id] - alpha
            original_weights[index_id] = alpha * original_weights[index_id]
            original_weights = original_weights / original_weights.sum()  # 归一化
            error_std = max(std_result) - min(std_result)

            iter_count += 1

        return list(original_weights), iter_count

    # 中心权重函数
    @staticmethod
    def center_weights_func(trains_df, test_without_label_seris, center_vector_df, k_values=1, alpha=0.5):
        trains_sets_weights = trains_df[
            [i for i in trains_df.columns if 'col' in i]]
        trains_label = trains_df['label']
        trains_mean_value = trains_df['mean']

        # 计算距离
        tmp_dis = (abs((test_without_label_seris *
                        trains_sets_weights).sum(axis=1) - trains_mean_value)) ** 2
        tmp_dis = tmp_dis.apply(lambda x: np.sqrt(x))
        # # 排序 并获取前 k 个值
        # tmp_dis_sort_k = tmp_dis.sort_values()[ : k_values]
        # # 获取分类对应标签
        # result_label = list(trains_df.loc[tmp_dis_sort_k.index].label)
        # class_label_number_dict = {result_label.count(keys): keys for keys in list(set(trains_df.label))}

        # # 输出个数最多的一组
        # class_label_result = class_label_number_dict[max(class_label_number_dict.keys())]

        dict_result = {keys.split(
            ':')[0]: value for keys, value in tmp_dis.to_dict().items()}
        weights_series = pd.Series(dict_result)

        """添加算法, 中心距来衡量"""
        test_center_dis = test_without_label_seris - center_vector_df
        test_center_dis = test_center_dis.apply(
            lambda x: x * x).sum(axis=1)  # 求欧式距离
        test_center_dis = test_center_dis.apply(lambda x: np.sqrt(x))

        # 合并两个结果
        weights_series.index = test_center_dis.index
        weigt_tmp_df = weights_series.to_frame(name='weights_dis')
        center_tmp_df = test_center_dis.to_frame(name='center_dis')

        """对中心距进行归一化计算, 发现效果更佳: 对部分数据效果显著"""
        center_tmp_df = center_tmp_df / center_tmp_df.sum()

        # 可以进行改动的地方 现在用的算术平均 加权线性平均
        result = pd.concat([weigt_tmp_df * (1 - alpha),
                            center_tmp_df * (alpha)], axis=1).sum(axis=1).idxmin()

        return result

    # 添加代码时间 2017年11月9日

    # 计算同类别的中心
    def get_class_center_func(self, data_with_label_df, label_name = 'label'):
        # 获取列名 list
        tmp_class_name_list = data_with_label_df.columns
        columns_name_list = [i for i in values.columns if '{name}'.format(name = label_name) not in i]

        return data_with_label_df[columns_name_list].mean()



    """尚未优化的函数"""
    # 数据可视化
    # lines graph

    def get_2d_graph(self, x_data, y_data=None, style=None):
        '''
        本函数 作图数据要求为 df 格式
        针对各种图表进行编辑
        sytle : hist(默认), boxplot,  
        '''
        if y_data is None:  # 针对只有一个输入数据作图
            if style is None:
                x_data.hist()
                plt.show()
            elif style is 'boxplot':
                x_data.boxplot()
                plt.show()
            elif style is 'lines':
                x_data.plot()
                plt.grid(True)
                plt.show()
        else:
            plt.plot(x_data, y_data)
            plt.grid(True)
            plt.show()

    # KNN 算法
    '''
     |  ==============  ====================  ========  ===============================
     |  identifier      class name            args      distance function
     |  --------------  --------------------  --------  -------------------------------
     |  "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
     |  "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
     |  "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``
     |  "minkowski"     MinkowskiDistance     p         ``sum(|x - y|^p)^(1/p)``
     |  "wminkowski"    WMinkowskiDistance    p, w      ``sum(w * |x - y|^p)^(1/p)``
     |  "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
     |  "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
     |  ==============  ====================  ========  ===============================
    '''
    # @staticmethod

    def classic_knn_alg(self, train_array, train_label, test_array, test_label, k_value=1, weights='uniform', p_value=2, metric='minkowski', style=None):
        '''
        train_array train_label 为 ndarray 数据
        该算法是 KNN 算法, 仅介绍一些经典应用
        具体 or 更复杂的应用 请查阅 索引 相关的资料或程序

        输出 正确率, 样本与近邻训练集样本的距离 以及 与近邻训练集样本的索引位置
        '''
        from sklearn import neighbors as knn
        import time

        columns_name = self.set_columns_name()
        df = self.set_df_frame(columns=columns_name)
        if (style is None):
            start_time = time.clock()
            clf = knn.KNeighborsClassifier(
                n_neighbors=k_value, weights=weights, p=p_value)
            clf.fit(train_array, train_label)
            y_pred = clf.predict(test_array)
            end_time = time.clock()
            accuracy = self.get_accuracy_score(y_pred, test_label) * 100
            df.loc[0, :] = [accuracy, k_value, end_time - start_time]
            return df  # , clf.knn(test_array)

        elif (style is not None) and (style <= 20):
            for nn, ix in enumerate(range(k_value - 1, style + 1)):
                k_value = ix + 1
                start_time = time.clock()
                clf = knn.KNeighborsClassifier(
                    n_neighbors=k_value, weights=weights, p=p_value, metric=metric)
                clf.fit(train_array, train_label)
                y_pred = clf.predict(test_array)
                end_time = time.clock()
                accuracy = self.get_accuracy_score(y_pred, test_label) * 100
                # 存储
                df.loc[nn, :] = [accuracy, k_value, end_time - start_time]
            return df  # , clf.knn(test_array)
        else:
            return 'k 近邻算法, k 取值过大时, 效果并不理想, 建议输入区间为[2,20]的整数值.'

    # 改进的 KNN 算法
    """    
    该算法的使用是基于定义的函数来实现
    定义函数 有 improve_knn_alg 函数结合
    """
    @staticmethod
    def mydist(x, y, **kwargs):
        '''
        x,y 不可交换性 除非欧式距离
        x 为待分类数据
        y 为训练集数据
        '''
        return np.sum((x - y) * kwargs["power"] / kwargs["denominator"])

    @staticmethod
    def improve_knn_alg(x_arr_train, y_train_label, k_value=3, weights=mydist, metric=mydist, metric_dict={"power": 2, "denominator": 1}):
        '''
        输入: 
            np.array 数据
            定义的函数输出数据类型也以 np.array 为主
        输出: 
            基于当前距离下的 KNN 类, 
            nbrs.predict()预测属于类标签
            nbrs.predict_proba() 预测属于各个类的概率
            nbrs.kneighbors() 给出 测试样本与邻近样本的距离, 以及索引位置
        说明: 该算法是优势在于, 研究者可以任意定义各种距离函数, 从而实现运算. 从而突破传统的距离函数约束
        '''
        from sklearn import neighbors
        nbrs = neighbors.KNeighborsClassifier(
            n_neighbors=k_value, metric=metric, metric_params=metric_dict)
        nbrs.fit(x_arr_train, y_train_label)
        return nbrs
