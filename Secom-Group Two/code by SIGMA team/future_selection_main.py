# This Python file uses the following encoding: utf-8
"""
Contains functions to plot "heatmaps" of a Confusion Matrix ,"Roc_curve" and "PR_curve"
Using smote and GAN to Process imbalanced data
Author: Junjie He,Xin Liu,Liling Zuo
Created on Tuesday Apr 28, 2018
"""
from sklearn import preprocessing
from GAN_BP import GAN
import tensorflow as tf
from sklearn import metrics as mr
import pandas as pd
import numpy as np
import os
import numpy as np
from GAN_BP import GAN
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from feature_selection_ga import FeatureSelectionGA
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


imb_methond = "smote"
Gan_max_epoch = 300
GA_pop_num = 20
GA_pop_generation = 50
GA_cross_prob = 0.8
GA_mutate_prob = 0.4
Classifier_model = AdaBoostClassifier()
def discretization_data(X_in):
    row, column = X_in.shape
    d1 = np.zeros((row, column))
    for i in range(X_in.shape[1]):
        row = X_in[:, i]
        k = 10
        a = pd.cut(row, k, labels=range(k))
        d1[:, i] = a
    return d1

def balance_data(x, y, methond="smote"):
    if methond == "gan":
        imbdata = get_label_t(x, y, t=1)
        gan = GAN(imbdata, max_epoch=Gan_max_epoch, batch_size=imbdata.shape[0])
        data = gan.gan_data()
        y_gan = np.ones([data.shape[0], 1], dtype=int)
        x = np.concatenate((x, data), axis=0)
        y = np.concatenate((y, y_gan), axis=0)

    elif methond == "smote":
        sm = SMOTEENN()
        x, y = sm.fit_sample(x, y)
    else:
        print("The methond is invalid!")
    tmp_list = np.random.permutation(x.shape[0])
    x = x[tmp_list, :]
    y = y[tmp_list]
    return x, y

def calculate_information(x, y):
    each_information = []
    feature = []
    for row in range(x.shape[1]):
        each_information.append(mr.mutual_info_score(x[:, row], y))
    sort_information = sorted(each_information, reverse=True)
    x_bar = range(590)
    y_bar = each_information

    plt.xlabel('features')
    plt.ylabel('informations')
    plt.bar(x_bar, y_bar,width=5 ,color='rgb')
    plt.legend('features')
    plt.show()

    length = 0
    for i in range(100):
        for j in range(len(each_information)):
            if length < 100:
                if each_information[j] == sort_information[i]:
                    feature.append(j)
                    each_information[j] = None
                    length += 1
    x_selected = np.zeros((x.shape[0], len(feature)))
    for i in range(x.shape[0]):
        for j in range(len(feature)):
            x_selected[i][j] = x[i][feature[j]]
    y_selected = np.zeros((len(y), 1), dtype=int)
    for i in range(len(y)):
        y_selected[i][0] = y[i]

    return x_selected, y_selected,feature


def get_label_t(x, y, t):
    x_tmp = []
    for i in range(len(y)):
        if y[i] == t:
            x_tmp.append(x[i])
    x_tmp = np.array(x_tmp).reshape([-1, 100])
    return x_tmp


def print_last_feature(gene):
    feature_last = []
    for i in range(len(gene)):
        if gene[i] == 1:
            feature_last.append(features[i])
    print('The last features that we choose is')
    print(sorted(feature_last))

data_start = pd.read_csv('dataset.csv', header=None)
# print(data_start)
data_start = data_start.drop_duplicates(keep='first', inplace=False)
# print(data_start)
data = data_start.fillna(data_start.mean()).values


X, y,features = calculate_information(data[:, 0:590], data[:, 590])


min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


permutation = np.random.permutation(X.shape[0])
shuffled_dataset = X[permutation, :]
shuffled_labels = y[permutation]


num_train = 941
X_train = shuffled_dataset[0:num_train, :]
y_train = shuffled_labels[0:num_train]
num_test = 313
X_test = shuffled_dataset[num_train:num_train + num_test, :]
y_test = shuffled_labels[num_train:num_train + num_test]
X_development = shuffled_dataset[num_train + num_test:1567, :]
y_development = shuffled_labels[num_train + num_test:1567]


print(X_train.shape)
print("The number of training imbsamples:%d" % X_train.shape[0])
X_train, y_train = balance_data(X_train, y_train, methond=imb_methond)
print(X_train.shape)
print("The number of balanced training samples:%d" % X_train.shape[0])

fsga = FeatureSelectionGA(Classifier_model, x=X_train, y=y_train, x_test=X_test, y_test=y_test,
                          x_development=X_development, y_development=y_development)
pop = fsga.generate(n_pop=GA_pop_num, cxpb=GA_cross_prob, mutxpb=GA_mutate_prob, ngen=GA_pop_generation)
# Select the best individual from the final population and fit the initialized model
gene=fsga.best_ind
print_last_feature(gene)