import random
from sklearn.metrics import f1_score,precision_score,recall_score   #准确率，精确率，召回率
from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
from sklearn import preprocessing
import operator
from sklearn import metrics as mr   #计算互信息
from sklearn.ensemble import AdaBoostClassifier     #分类器




####适应度函数
class FitenessFunction:
    def __init__(self, n_splits=3, *args, **kwargs):
        """
            Parameters
            -----------
            n_splits :int,
                Number of splits for cv

            verbose: 0 or 1
        """
        self.n_splits = n_splits

    def calculate_fitness(self, model, x, y, x_test, y_test, x_development, y_development, ):      #计算适应度
        model.fit(x, y)  # 用训练数据拟合分类器模型，，总样本
        predicted_y_train = model.predict(x)  # 得到y训练的预测概率
        # print(predicted_y_train)
        predicted_y_test = model.predict(x_test)  # 得到y测试的预测概率
        predicted_y_development = model.predict(x_development)  # #得到y验证的预测概率

        f1_score_train = f1_score(y, predicted_y_train)  # 得到准确率
        f1_score_test = f1_score(y_test, predicted_y_test)
        f1_score_development = f1_score(y_development, predicted_y_development)
        precision_score_test = precision_score(y_test, predicted_y_test)  # 得到精确率
        precision_score_development = precision_score(y_development, predicted_y_development)
        recall_score_test = recall_score(y_test, predicted_y_test)  # 得到召回率
        recall_score_development = recall_score(y_development, predicted_y_development)

        train_acc = model.score(x, y)       #
        test_acc = model.score(x_test, y_test)      #
        develop_acc = model.score(x_development, y_development)         #

        print('acc')
        print(train_acc, '     ', test_acc, '     ', develop_acc)
        print('f1_score:准确率')
        print(f1_score_train, '     ', f1_score_test, '     ', f1_score_development)
        print('precision_score:精确率')
        print(precision_score_test, '     ', precision_score_development)
        print('recall_score:召回率')
        print(recall_score_test, '     ', recall_score_development)

        return recall_score_test
###特征选择
class GA(object):

    def __init__(self, aCrossRate, aMutationRage,  aGeneLenght, atournamentSize,subalgebra,individual_number,
                 X_train,y_train,X_test,y_test,X_verify,y_verify):
        self.crossRate = aCrossRate  # 交叉概率 #
        self.mutationRate = aMutationRage  # 突变概率 #
        self.geneLenght = aGeneLenght  # 基因长度 #
        self.tournamentSize = atournamentSize  # 竞标赛选取个数 #
        self.subalgebra = subalgebra        #子代数
        self.individual_number = individual_number      #个体数

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_verify = X_verify
        self.y_verify = y_verify

        self.best = []  # 保存这一代中最好的个体

        self.total_gene=np.zeros((self.individual_number,self.geneLenght))  #存放每一代个体的基因

        self.total_gene_next=np.zeros((self.individual_number,self.geneLenght))


        self.Run1()

    def Run1(self):    #循环子代
        for i in range(self.subalgebra):             #每个子代都要循环一次
            print('第{0}代'.format(i))

            if i==0:
                self.initialize()
            else:
                self.Run2()
        # print(self.best)
        max=sorted(self.best)
        print(max[-1])

            # print(self.best)


    def initialize(self):     #循环个体，将每个个体对应的特征找出来，0舍去，1获取
        best_1=[]
        bestMatchValue = []  # 记录每代中所有个体的适应值
        for j in range(self.individual_number):
            gene = np.random.randint(0, 2, 100)     #基因为随机的0,1，共100个
            self.total_gene[j,:]=gene          #将基因存放进去
            # print("00000000000000000000000")
            # print(self.total_gene)
            new_X_train=np.zeros((self.X_train.shape[0],np.sum(gene)))    #初始化
            new_X_test=np.zeros((self.X_test.shape[0],np.sum(gene)))
            new_X_verify=np.zeros((self.X_verify.shape[0],np.sum(gene)))
            m=0
            for k in range(100):
                if gene[k]==0:
                    continue
                else:                                   #将对应基因的特征选取
                    new_X_train[:,m]=self.X_train[:,k]
                    new_X_test[:,m]=self.X_test[:,k]
                    new_X_verify[:,m]=self.X_verify[:,k]
                    m+=1
            # print(new_X_train.shape)
            fitness=self.Run3(new_X_train,new_X_test,new_X_verify)  #得到适应度值
            bestMatchValue.append(fitness)     #每一个个体的适应度值
            # self.total_gene[i,:]=gene           #将每一代个体的基因存储起来
        p1, p2 = self.judge(bestMatchValue,best_1)  # 评估

        for i in range(self.individual_number):      #生成新的子代
            next_gene=self.newChild(p1,p2)
            self.total_gene_next[i,:]=next_gene              #更新子代

    def Run2(self):
        best_1=[]
        bestMatchValue = []  # 记录每代中所有个体的适应值
        for j in range(self.individual_number):

            wide=int(np.sum(self.total_gene_next[j]))    #选择基因的长度
            # print("00000000000000000000000")
            # print(self.total_gene)
            new_X_train=np.zeros((self.X_train.shape[0],wide))    #初始化
            new_X_test=np.zeros((self.X_test.shape[0],wide))
            new_X_verify=np.zeros((self.X_verify.shape[0],wide))
            m=0
            for k in range(100):
                if self.total_gene_next[j][k]==0:
                    continue
                else:                                   #将对应基因的特征选取
                    new_X_train[:,m]=self.X_train[:,k]
                    new_X_test[:,m]=self.X_test[:,k]
                    new_X_verify[:,m]=self.X_verify[:,k]
                    m+=1
            # print(new_X_train.shape)
            fitness=self.Run3(new_X_train,new_X_test,new_X_verify)  #得到适应度值
            bestMatchValue.append(fitness)     #每一个个体的适应度值
            # self.total_gene[i,:]=gene           #将每一代个体的基因存储起来
        p1, p2 = self.judge(bestMatchValue,best_1)  # 评估

        for i in range(self.individual_number):      #生成新的子代
            next_gene=self.newChild(p1,p2)
            self.total_gene_next[i,:]=next_gene              #更新子代

    def Run3(self,X_train,X_test,X_verify):     #每个个体需要循环GA，训练模型
        model = AdaBoostClassifier(n_estimators=100)        #分类器
        FF=FitenessFunction()       #调用适应度函数
        fitness=FF.calculate_fitness(model,X_train,self.y_train,X_test,self.y_test,X_verify,self.y_verify)
        return fitness      #返回这个个体的适应值（召回率）


    def judge(self,bestMatchValue,best_1):
        """评估，计算每一个个体的适应值，记录最好值及个体,选择并返回父代1和父代2"""
        gene_idx=list(range(self.individual_number))      #建立基因下标
        compound_index_fitness=dict(zip(gene_idx,bestMatchValue))      #建立基因字典
        # print(compound_index_fitness)
        sorted_CIF = dict(sorted(compound_index_fitness.items(), key=operator.itemgetter(1)))  # 对字典进行排序,从小到大
        gene_index = list(sorted_CIF.keys())
        # print(gene_index)
        max_gene_index=gene_index[-1]       #最大适应值的基因的下标

        parent1=self.total_gene[max_gene_index,:]              #父代1
        parent2=self.total_gene[gene_index[-2],:]              #父代2
        # print(parent1,parent2)
        fitness_index=list(sorted_CIF.values())
        print(fitness_index)
        max_fitness=fitness_index[-1]

        best_1.append(max_fitness)      #每一代中最好的适应度
        self.best.append(max_fitness)   #求总的最好适应度

        return parent1,parent2

    def order_Based_cross(self,parent1,parent2):
        '''  Order-Based Crossover，随机排序交叉算子'''
        newgene=parent1[:]
        son=list(range(self.geneLenght))
        for i in range(0,self.geneLenght):
            son[i]=None
        random_data=random.randint(0,self.geneLenght-1)
        random_set=list(range(random_data))
        for i in range(0,random_data):
            random_set[i]=random.randint(0,self.geneLenght-1)
        for x in random_set:
            son[x]=parent2[x]
        for x in range(len(son)):
            if son[x]==None:
                son[x]=newgene[x]

        return son

    def mutation(self, gene):
        """突变"""
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(0, self.geneLenght - 1)
        # 随机选择两个位置的基因交换--变异 #
        newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        return newGene

    def newChild(self,parent1,parent2):
        """产生单个后代"""
        #parent1 = self.getOne()
        # parent1 = self.tournamentSelect()
        rate = random.random()

        # 按概率交叉 #
        if rate < self.crossRate:
            # 交叉 #
            # parent2 = self.tournamentSelect()
            gene = self.order_Based_cross(parent1, parent2)
        else:
            gene = parent1

        # 按概率突变 #
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return gene

#导入数据集并去重
original_data=pd.read_csv('secom.csv',header=None)
original_data=original_data.drop_duplicates()
Data=np.array(original_data.values)

original_label=pd.read_csv('secom_Labels.csv',header=None)
Label=np.array(original_label.iloc[:,0])            #Label为列表，1x1576

# 异常值处理，功能是将NaN替换成所在列的均值。
data_mean = np.nanmean(Data, axis=0)  # 求跳过了NaN之后，所在列的均值的方法
for i in range(Data.shape[1]):  # 遍历第0列到最后一列
    temp = Data[:,i]
    temp[np.isnan(temp)] = data_mean[i]  # 用每一列剃掉NaN后的均值来替换NaN

#利用sklearn 实现归一化
min_max_scaler = preprocessing.MinMaxScaler()
Data_normalization=min_max_scaler.fit_transform(Data)

#离散化，等宽离散化
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

#计算互信息，并取出前100个

Data_mutual_information=list(range(Data.shape[1]))
for i in range(Data.shape[1]):
    mutual=Data_normalization[:,i]
    Data_mutual_information[i]=mr.mutual_info_score(Label,mutual)
'''print(Data_mutual_information)'''
mutual_information_index=list(range(590))
mutual_information_dict=dict(zip(mutual_information_index,Data_mutual_information))
sorted_x=dict(sorted(mutual_information_dict.items(),key=operator.itemgetter(1)))   #对字典进行排序
mutual_index=list(sorted_x.keys())

mutual_index_data=list(range(100))
for i in range(100):
    mutual_index_data[i]=mutual_index[-(i+1)]
'''print(mutual_index_data)'''

#取出100行数据,特征和标签都要取出来
Data_mutual=np.zeros((Data.shape[0],100))
j=0
for i in mutual_index_data:
    Data_mutual[:,j]=Data_normalization[:,i]
    j+=1
Label_mutual=np.zeros((100,1))
k=0
for i in mutual_index_data:
    Label_mutual[k,:]=Label[i]
    k+=1
'''print(Label_mutual)'''
'''print(Data_mutual)'''

# 随机打乱数据集
permutation = np.random.permutation(Data_mutual.shape[0])
Data_mutual_shuffled= Data_mutual[permutation, :]
Label_shuffled = Label[permutation]
# print(Data_mutual_shuffled)
# print(Label_shuffled)


#划分训练集，验证集，测试集(比例：5:2:3）
X_train=Data_mutual[:784]            #784x100
y_train=Label[:784]        #1x784
'''print(y_train)'''

X_verify=Data_mutual[785:1097]
y_verify=Label[785:1097]


X_test = Data_mutual[1097:]
y_test = Label[1097:]


#训练集平衡化smote并对数据样本进行合成
sm = SMOTEENN()    #random_state=42
X_res, y_res = sm.fit_sample(X_train,y_train)
# print(X_res)
# print(y_res.shape)
tmp_list = np.random.permutation(X_res.shape[0])  #随机打乱数据集
X_train_s = X_res[tmp_list, :]
y_train_s = y_res[tmp_list]
# print(X_train_s.shape)
# print(y_train_s.shape)




#分类评价
model = AdaBoostClassifier(n_estimators=100)
RG=GA(0.95, 0.02, 100,5,20,15,X_train_s,y_train_s,X_test,y_test,X_verify,y_verify)
