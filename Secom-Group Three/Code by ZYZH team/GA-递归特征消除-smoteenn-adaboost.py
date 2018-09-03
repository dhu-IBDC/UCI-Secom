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



    # ## 特征提取
    # estimator = LinearSVC()
    # selector = RFE(estimator=estimator, n_features_to_select=100)
    # X_t = selector.fit_transform(Data_normalization, Label)
    # dataframe = pd.DataFrame(X_t)
    # dataframe.to_csv("train_x.csv", index=False, header=None, sep=',')
    # dataframe = pd.DataFrame(Label)
    # dataframe.to_csv("train_y.csv", index=False, header=None, sep=',')

    Data_mutual=pd.read_csv('train_x.csv', header=None)
    Data = np.array(Data_mutual.values)

    original_label = pd.read_csv('train_y.csv', header=None)
    Label = np.array(original_label.iloc[:, 0])



    # 随机打乱数据集
    permutation = np.random.permutation(Data_mutual.shape[0])
    Data_mutual_shuffled = Data[permutation, :]
    Label_shuffled = Label[permutation]
    # print(Data_mutual_shuffled)
    # print(Label_shuffled)


    # # 划分训练集，验证集，测试集(比例：5:2:3）
    X_train = np.array(Data_mutual[:784] ) # 784x100
    y_train = Label[:784]  # 1x784
    print(X_train)
    print(X_train.shape)
    print(y_train.shape)

    X_valid = np.array(Data_mutual[785:1097])
    y_valid = Label[785:1097]

    X_test = np.array(Data_mutual[1097:])
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
    ga.run(50)
    ga.plot()
    os.system('pause')
