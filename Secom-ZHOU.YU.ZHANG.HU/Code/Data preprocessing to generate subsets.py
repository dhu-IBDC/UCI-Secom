import random
import os
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
# from data import get_data
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn import preprocessing
import operator
from sklearn import metrics as mr   #计算互信息
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
if __name__ == '__main__':
    # 读取数据
    # 导入数据集并去重
    original_data = pd.read_csv('secom.csv', header=None)
    original_data = original_data.drop_duplicates()
    Data = np.array(original_data.values)

    original_label = pd.read_csv('secom_Labels.csv', header=None)
    Label = np.array(original_label.iloc[:, 0])  # Label为列表，1x1576

    # 异常值处理，功能是将NaN替换成所在列的均值。
    data_mean = np.nanmean(Data, axis=0)  # 求跳过了NaN之后，所在列的均值的方法
    for i in range(Data.shape[1]):  # 遍历第0列到最后一列
        temp = Data[:, i]
        temp[np.isnan(temp)] = data_mean[i]  # 用每一列剃掉NaN后的均值来替换NaN
        # temp[np.isnan(temp)] = 0 # 用每一列剃掉NaN后的均值来替换NaN
    # 利用sklearn 实现归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    Data_normalization = min_max_scaler.fit_transform(Data)


    # all_X = VarianceThreshold(threshold=0.2).fit_transform(Data_normalization)
    # svc = SVC(kernel="linear", C=1)
    #
    # all_X, all_y = RFE(estimator=svc, n_features_to_select=200, step=1).fit_transform(Data_normalization, Label)
    # all_X_1 = PCA(n_components=100).fit_transform(all_X)
    # dataframe = pd.DataFrame(all_X_1)
    # dataframe.to_csv("train_x_1.csv",index=False,header=None,sep=',')
    # dataframe = pd.DataFrame(all_y)
    # dataframe.to_csv("train_y_1.csv",index=False,header=None,sep=',')

    ## 特征提取
    estimator = LinearSVC()
    selector = RFE(estimator=estimator, n_features_to_select=100)
    X_t = selector.fit_transform(Data_normalization, Label)
    dataframe = pd.DataFrame(X_t)
    dataframe.to_csv("train_x_1.csv", index=False, header=None, sep=',')
    dataframe = pd.DataFrame(Label)
    dataframe.to_csv("train_y_1.csv", index=False, header=None, sep=',')