# from data import get_data
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import metrics as mr   #计算互信息

# 计算互信息
def calculate_information(x, y):
    each_information = []
    feature = []
    for row in range(x.shape[1]):
        each_information.append(mr.mutual_info_score(x[:, row], y))
    sort_information = sorted(each_information, reverse=True)
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

#离散化数据
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
    if methond == "smoteenn":
        x, y = SMOTEENN().fit_sample( x, y)
    elif methond=="adasyn":
        x,y=ADASYN().fit_sample(x,y)
    elif methond=="smote":
        x, y = SMOTE(kind='borderline1').fit_sample(x, y)
    else:
        print("The methond is invalid!")
    tmp_list = np.random.permutation(x.shape[0])
    x = x[tmp_list, :]
    y = y[tmp_list]
    return x, y

def Classifier(methond="AdaBoostClassifier"):
    if methond=="AdaBoostClassifier":
        model = AdaBoostClassifier(n_estimators=100)
        return model
    elif methond=="KNeighborsClassifier":
        model = KNeighborsClassifier()
        return model
    elif methond=="RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=8)
        return model
    elif methond=="DecisionTreeClassifier":
        model = tree.DecisionTreeClassifier()
        return model
    else:
        print("The methond is invalid!")




# 获取训练集中t类样本
def get_label_t(x, y, t):
    x_tmp = []
    for i in range(len(y)):
        if y[i] == t:
            x_tmp.append(x[i])
    x_tmp = np.array(x_tmp).reshape([-1, 100])
    return x_tmp

#输出最后选取的特征
def print_last_feature(gene):
    feature_last = []
    for i in range(len(gene)):
        if gene[i] == 1:
            feature_last.append(features[i])
    print('The last features that we choose is')
    print(sorted(feature_last))

data_start = pd.read_csv('dataset.csv', header=None)
# print(data_start)
# 去重
data_start = data_start.drop_duplicates(keep='first', inplace=False)
# print(data_start)
# 填充缺失值
data = data_start.fillna(data_start.mean()).values

# 筛选互信息前100名
X, y,features = calculate_information(data[:, 0:590], data[:, 590])

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# 随机打乱数据集
permutation = np.random.permutation(X.shape[0])
shuffled_dataset = X[permutation, :]
shuffled_labels = y[permutation]

# 划分训练集测试集验证集
num_train = 941
X_train = shuffled_dataset[0:num_train, :]
y_train = shuffled_labels[0:num_train]
num_test = 313
X_test = shuffled_dataset[num_train:num_train + num_test, :]
y_test = shuffled_labels[num_train:num_train + num_test]
X_valid = shuffled_dataset[num_train + num_test:1567, :]
y_valid = shuffled_labels[num_train + num_test:1567]

# X_train, y_train = balance_data(X_train, y_train, methond="smote")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=8)
# X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=8)

# smote1 = SMOTE(random_state=8)
# data1, label1 = smote1.fit_sample(X_train, y_train)

smote2 = SMOTEENN(random_state=8)
data2, label2 = smote2.fit_sample(X_train, y_train)
data1,label1=smote2.fit_sample(X_train, y_train)
data3,label3=smote2.fit_sample(X_train, y_train)
# smote3 = ADASYN(random_state=8)
# data3, label3 = smote3.fit_sample(X_train, y_train)
def recall(gene):
    gene = np.array(gene).astype(int)
    index = np.where(gene==1)
    X_train_1 = data1[:, index[0]]
    X_train_2 = data2[:, index[0]]
    X_train_3 = data3[:, index[0]]
    X_test_ = X_test[:, index[0]]
    X_valid_= X_valid[:, index[0]]

    clf1 = AdaBoostClassifier(random_state=8)
    clf2 = KNeighborsClassifier()
    clf3 = RandomForestClassifier()

    clf1.fit(X_train_1, label1)
    clf2.fit(X_train_2, label2)
    clf3.fit(X_train_3, label3)

    y_prob = clf1.predict_proba(X_test_)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6.5))
    plt.plot(fpr, tpr, color='red', lw=2, linestyle=':',
                     label='AdaBoostClassifier (area = {0:0.2f})'
                     ''.format(roc_auc))

    y_prob = clf2.predict_proba(X_test_)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, linestyle='-',
             label='KNeighborsClassifier (area = {0:0.2f})'
                   ''.format(roc_auc))

    y_prob = clf3.predict_proba(X_test_)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='green', lw=2, linestyle='-.',
             label='RandomForestClassifier (area = {0:0.2f})'
                   ''.format(roc_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for Baseline, Weighted and CV', fontsize=14)
    plt.legend(title='AUC', loc="lower right")
    plt.show()

gene1 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]

recall(gene1)
