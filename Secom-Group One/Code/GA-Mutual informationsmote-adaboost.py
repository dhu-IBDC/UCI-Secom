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
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from  sklearn import model_selection



TIME_NONE = -1
class Life(object):
    """个体类"""

    def __init__(self, aGene=None):
        self.gene = aGene
        # self.time = 0  # 初始化时间 #
        self.recall = 0
        self.conf_mat_test = None
        self.conf_mat_valid = None

    def __len__(self):
        return len(self.gene)

    def __getitem__(self, item):
        return self.gene[item]


class GA(object):
    """遗传算法类"""

    def __init__(self,
                 amatchFun,
                 aCrossRate,
                 aMutationRage,
                 aLifeCount,
                 aGeneLenght,
                 atournamentSize):
        self.matchFun = amatchFun  # 适应性函数 #
        self.crossRate = aCrossRate  # 交叉概率 #
        self.mutationRate = aMutationRage  # 突变概率 #
        self.lifeCount = aLifeCount   # 个体数 #
        self.geneLenght = aGeneLenght  # 基因长度 #
        self.tournamentSize = atournamentSize  # 竞标赛选取个数 #

        self.lives = []  # 种群 #
        self.best = None  # 保存这一代中最好的个体 #
        self.bestMatchValue = []  # 记录每一代中最好的结果 #
        self.bounds = 0.0  # 适配值之和，用于选择时计算概率 #
        self.initPopulation()  # 初始化种群 #

    def run(self, n):
        # 开始迭代
        for i in range(n):
            # 评估当前种群
            self.judge()
            # 进化种群
            self.evolution()
            os.system('cls')
            print('迭代次数:', i+1)
            print('最适配个体:', self.best.gene)
            print('最适配值:', self.best.recall)
            print('测试集混淆矩阵:\n',self.best.conf_mat_test)
            print('验证集混淆矩阵:\n',self.best.conf_mat_valid)
            # print('\r迭代次数：{0}\n最适配个体：{1}\n最适配值：{2}'.format(i+1, self.best.gene, self.matchFun(self.best)))
        print('*' * 30)
        print('final result:')
        print('个体数:', self.lifeCount)
        print('基因长度:', self.geneLenght)
        print('交叉概率:', self.crossRate)
        print('突变概率:', self.mutationRate)
        print('锦标赛规模:', self.tournamentSize)
        print('最适配个体:', self.best.gene)
        print('最适配值:', self.matchFun(self.best))
        print('*' * 30)

    # def initPopulation(self):
    #     """初始化种群"""
    #     for i in range(self.lifeCount):
    #         gene = [x for x in range(self.geneLenght)]
    #         random.shuffle(gene)  # 随机洗牌 #
    #         life = Life(gene)
    #         self.lives.append(life)

    def initPopulation(self):
        """初始化种群"""
        for i in range(self.lifeCount):     #循环个体数
            gene = list(np.random.randint(0, 2, size=self.geneLenght))
            life = Life(gene)
            self.lives.append(life)

    # def judge(self):
    #     """评估，计算每一个个体的适配值，记录最好值及个体"""
    #     self.bounds = 0.0
    #     self.best = self.lives[0]
    #     for life in self.lives:
    #         life.time = self.matchFun(life)
    #         self.bounds += life.time
    #         if self.best.time > life.time:   # time越小越好 #
    #             self.best = life
    #     self.bestMatchValue.append(self.best.time)
    def judge(self):
        """评估，计算每一个个体的适配值，记录最好值及个体"""
        self.bounds = 0.0
        self.best = self.lives[0]
        for life in self.lives:
            life.recall, life.conf_mat_test, life.conf_mat_valid= self.matchFun(life)
            self.bounds += life.recall
            if self.best.recall < life.recall:   # time越小越好 #
                self.best = life
        self.bestMatchValue.append(self.best.recall)

    def cross(self, parent1, parent2):  #交叉
        start = random.randint(0, self.geneLenght - 1)  # 随机生成突变起始位置 #
        end = random.randint(0, self.geneLenght - 1)  # 随机生成突变终止位置 #
        childGene = [None for i in range(self.geneLenght)]
        childGene[start:end] = parent2[start:end]
        childGene[:start] = parent1[:start]
        childGene[end:] = parent1[end:]
        return childGene

    def order_Based_cross(self, parent1, parent2):
        '''  Order-Based Crossover，随机排序交叉算子'''
        newgene = parent1[:]
        son = list(range(self.geneLenght))
        for i in range(0, self.geneLenght):
            son[i] = None
        random_data = random.randint(0, self.geneLenght - 1)
        random_set = list(range(random_data))
        for i in range(0, random_data):
            random_set[i] = random.randint(0, self.geneLenght - 1)
        for x in random_set:
            son[x] = parent2[x]
        for x in range(len(son)):
            if son[x] == None:
                son[x] = newgene[x]

        return son


    def mutation(self, gene):
        newGene = gene[:]
        position = np.random.randint(0, self.geneLenght-1)
        newGene[position] = 1-gene[position]
        return newGene


    # def orderCross(self, parent1, parent2):
    #     """
    #     函数功能：交叉
    #     函数实现：随机交叉长度为n的片段，n为随机产生
    #     """
    #     start = random.randint(0, self.geneLenght - 1)  # 随机生成突变起始位置 #
    #     end = random.randint(0, self.geneLenght - 1)  # 随机生成突变终止位置 #
    #     childGene = [None for i in range(self.geneLenght)]
    #     # 保留基因段
    #     if start > end:
    #         for i in range(end, start):
    #             childGene[i] = parent2[i]
    #     elif end > start:
    #         for i in range(start, end):
    #             childGene[i] = parent2[i]
    #     else:
    #         return parent1.gene
    #
    #     for i in range(self.geneLenght):
    #         if not parent1[i] in childGene:
    #             for j in range(self.geneLenght):
    #                 if childGene[j] is None:
    #                     childGene[j] = parent1[i]
    #                     break
    #     return childGene
        # p1len = 0
        # for g in parent1.gene:
        #     if p1len == index1:
        #         newGene.extend(tempGene)  # 插入基因片段
        #         p1len += 1
        #     if g not in tempGene:
        #         newGene.append(g)
        #         p1len += 1
        # self.crossCount += 1

        # return newGene

    # def positionBasedCrosee(self, parent1, parent2):
    #     # 随机选择个数
    #     num = np.random.randint(0, self.geneLenght)
    #     # 随机选择位置
    #     positonList = np.random.randint(0, self.geneLenght, size=num)
    #     positonList = list(set(positonList))  # 去重
    #
    #     childGene = [None for i in range(self.geneLenght)]
    #
    #     for position in positonList:
    #         childGene[position] = parent2[position]
    #
    #     for i in range(self.geneLenght):
    #         if not parent1[i] in childGene:
    #             for j in range(self.geneLenght):
    #                 if childGene[j] is None:
    #                     childGene[j] = parent1[i]
    #                     break
    #     return childGene

    # def mutation(self, gene):
    #     """突变"""
    #     index1 = random.randint(0, self.geneLenght - 1)
    #     index2 = random.randint(0, self.geneLenght - 1)
    #     #index3 = random.randint(0, self.geneLenght - 1)
    #     #index4 = random.randint(0, self.geneLenght - 1)
    #     # 随机选择两个位置的基因交换--变异 #
    #     newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群
    #     newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
    #     #newGene[index3], newGene[index4] = newGene[index4], newGene[index3]
    #     return newGene

    def rouletteSelect(self):
        """轮盘赌选择算子"""
        r = random.uniform(0, self.bounds)
        for life in self.lives:
            r -= life.time
            if r <= 0:
                return life
        raise Exception("选择错误", self.bounds)

    def tournamentSelect(self):
        '''竞标赛选择算子'''
        tournamentLives = []
        for i in range(self.tournamentSize):
            tournamentLives.append(random.choice(self.lives))
        tournamentLives = sorted(tournamentLives, key=lambda x: x.recall, reverse=True)
        return tournamentLives[0]

    def newChild(self):
        """产生单个后代"""
        #parent1 = self.getOne()
        parent1 = self.tournamentSelect()
        rate = random.random()
        # 按概率交叉 #
        if rate < self.crossRate:
            # 交叉 #
            parent2 = self.tournamentSelect()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene
        # 按概率突变 #
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)
        return Life(gene)

    def evolution(self):
        """产生下一代, 更新lives"""
        newLives = []
        newLives.append(self.best)  # 把最好的个体加入下一代 #
        while len(newLives) < self.lifeCount:
            newLives.append(self.newChild())
        self.lives = newLives

    def plot(self):
        x = range(len(self.bestMatchValue))
        y = self.bestMatchValue
        plt.plot(x, y)
        plt.ylabel('total ready time')
        plt.xlabel('iters')
        plt.show()


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

    # 离散化，等宽离散化
    # def dataDiscretize(dataSet):
    #     m,n = np.shape(dataSet)    #获取数据集行列（样本数和特征数)
    #     disMat = np.tile([0],np.shape(dataSet))  #初始化离散化数据集
    #     for i in range(n):
    #         x = [l[i] for l in dataSet] #获取第i+1特征向量
    #         y = pd.cut(x,10,labels=[0,1,2,3,4,5,6,7,8,9])   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类
    #         # print(y)
    #         for k in range(m):
    #             disMat[k][i] = y[k] #将离散化值传入离散化数据集
    #     return disMat
    # Data_discretization=dataDiscretize(Data_normalization)

    # 计算互信息，并取出前100个

    Data_mutual_information = list(range(Data.shape[1]))
    for i in range(Data.shape[1]):
        mutual = Data_normalization[:, i]
        Data_mutual_information[i] = mr.mutual_info_score(Label, mutual)
    '''print(Data_mutual_information)'''
    mutual_information_index = list(range(590))
    mutual_information_dict = dict(zip(mutual_information_index, Data_mutual_information))
    sorted_x = dict(sorted(mutual_information_dict.items(), key=operator.itemgetter(1)))  # 对字典进行排序
    mutual_index = list(sorted_x.keys())

    mutual_index_data = list(range(100))
    for i in range(100):
        mutual_index_data[i] = mutual_index[-(i + 1)]
    '''print(mutual_index_data)'''

    # svc = SVC(kernel="linear", C=1)
    # all_X, all_y = RFE(estimator=svc, n_features_to_select=100, step=1).fit_transform(Data_normalization, Label)
    # all_X = PCA(n_components=100).fit_transform(all_X)

    #
    # ## 特征提取
    # estimator = LinearSVC()
    # selector = RFE(estimator=estimator, n_features_to_select=100)
    # X_t = selector.fit_transform(Data_normalization, Label)
    # dataframe = pd.DataFrame(X_t)
    # dataframe.to_csv("train_x.csv", index=False, header=None, sep=',')
    # dataframe = pd.DataFrame(Label)
    # dataframe.to_csv("train_y.csv", index=False, header=None, sep=',')
    #
    # Data_mutual=



    # 取出100行数据,特征和标签都要取出来
    Data_mutual = np.zeros((Data.shape[0], 100))
    j = 0
    for i in mutual_index_data:
        Data_mutual[:, j] = Data_normalization[:, i]
        j += 1
    Label_mutual = np.zeros((100, 1))
    k = 0
    for i in mutual_index_data:
        Label_mutual[k, :] = Label[i]
        k += 1
    '''print(Label_mutual)'''
    '''print(Data_mutual)'''

    # 随机打乱数据集
    permutation = np.random.permutation(Data_mutual.shape[0])
    Data_mutual_shuffled = Data_mutual[permutation, :]
    Label_shuffled = Label[permutation]
    # print(Data_mutual_shuffled)
    # print(Label_shuffled)


    # # 划分训练集，验证集，测试集(比例：5:2:3）
    X_train = Data_mutual[:784]  # 784x100
    y_train = Label[:784]  # 1x784
    '''print(y_train)'''

    X_valid = Data_mutual[785:1097]
    y_valid = Label[785:1097]

    X_test = Data_mutual[1097:]
    y_test = Label[1097:]

    # X_train, X_test, y_train, y_test = train_test_split(Data_mutual_shuffled, Label_shuffled , test_size=0.4, random_state=8)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=8)
    # 过采样
    smote = SMOTEENN(random_state=8)
    data, label = smote.fit_sample(X_train, y_train)

    geneLength = X_train.shape[1]

    def recall_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        # mcc = matthews_corrcoef(y_true, y_pred)
        recall1 = (float(cm[1][1]) / np.sum(cm[1]))
        return recall1

    def matchFun(gene):
        gene = np.array(gene).astype(int)
        index = np.where(gene == 1)
        X_train_select = data[:, index[0]]
        X_test_select = X_test[:, index[0]]
        X_valid_select = X_valid[:, index[0]]

        clf = AdaBoostClassifier(random_state=8)        #分类器
        clf.fit(X_train_select, label)
        pre_test = clf.predict(X_test_select)
        pre_valid = clf.predict(X_valid_select)
        val = recall_score(y_test, pre_test)
        # print(confusion_matrix(y_test, pre))
        return val, confusion_matrix(y_test, pre_test), confusion_matrix(y_valid, pre_valid)


    ga = GA(matchFun, 0.95, 0.05, 100, 100, 2)
    ga.run(10)
    ga.plot()
    os.system('pause')
