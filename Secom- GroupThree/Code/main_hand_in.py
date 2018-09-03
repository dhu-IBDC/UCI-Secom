#this is the main program
#2018.8.22
import pandas as pd
import numpy as np
from sklearn import metrics as mtr
import operator
from imblearn.combine import SMOTEENN
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from feature_selection_ga import FeatureSelectionGA
import time


#read the dataset
FILE_PATH = r'C:\\Users\\Administrator\\PycharmProjects\\secompro\\dataset.csv'
original_data = pd.read_csv(FILE_PATH,header=None)



#caculate mutual information
def huxinxijisuan(X,y):


    hu_results = []
    for i in range(X.shape[1]):
        x_single = X[:,i]

        hu_result = mtr.mutual_info_score(y,x_single)
        hu_results.append(hu_result)

    list1 = list(range(590))
    middle1 = dict(zip(list1, hu_results))
    middle2 = dict(sorted(middle1.items(), key=operator.itemgetter(1)))

    print(list(middle2.keys()))
    jian = list(middle2.keys())
    tanlan_results = []
    for i in range(100):
        tanlan_results.append(jian[-(1 + i)])

    x_data = np.zeros((X_tezheng.shape[0], 100))
    j = 0
    for i in tanlan_results:
        x_data[:, j] = X_tezheng[:, i]
        j += 1


    return x_data,  jian



#normalized processing
def guiyi(X):

    for i in range(X.shape[1]):
        Xmax = np.max(X[:,i])
        Xmin = np.min(X[:,i])
        for j in range(X.shape[0]):
            if (Xmax - Xmin) == 0:
                X[j][i] = 1
            else:
                X[j][i]=(X[j][i]-Xmin)/(Xmax-Xmin)

    return X
print('lines of original dataset',original_data.shape[1])

#the first step: deleta duplicate lines
original_sepecific = original_data.drop_duplicates(keep='first', inplace=False)
print('lines of dataset after delete the duplicate lines',original_sepecific.shape[1])

start_time = time.time()

#the second step: remove NaN counts
original_clean = np.array(original_sepecific.values)
data_mean = np.nanmean(original_clean, axis=0)
for i in range(original_clean.shape[1]):
     temp = original_clean[:,i]
     temp[np.isnan(temp)] = data_mean[i]

#the third step: calculate mutual information
#before mutual information calculation, feature discretization first
X_tezheng = original_clean[:, 0:590]
X_ls = np.empty((X_tezheng.shape[0], X_tezheng.shape[1]), dtype=int)
for i in range(X_ls.shape[1]):

    k = 2
    X_ls[:, i] = pd.cut(X_tezheng[:, i], k, labels=range(k))
print('lisanhua',X_ls)

#call the function, to calculate the mutual information, and then get top 100 results
X, feature = huxinxijisuan(X_ls, original_clean[:, 590])
y = original_clean[:, 590]



#the forth step: normalized processing
X_guiyi_result = guiyi(X)
print('guiyihuahoude',X_guiyi_result)
print(X_guiyi_result.shape)

#the fifth step: disorder the new feature by lines
permutation_way = np.random.permutation(X_guiyi_result.shape[0])
shuffled_dataset = X_guiyi_result[permutation_way, :]
shuffled_labels = y[permutation_way]

#the sixth step: dividing dataset
num_train = 941
X_train = shuffled_dataset[0:num_train, :]
y_train = shuffled_labels[0:num_train]
num_test = 313
X_test = shuffled_dataset[num_train:num_train + num_test, :]
y_test = shuffled_labels[num_train:num_train + num_test]
X_development = shuffled_dataset[num_train + num_test:1567, :]
y_development = shuffled_labels[num_train + num_test:1567]

#the seventh step: balanced processing

#verification the shape first
print('training ',X_guiyi_result.shape)
print('labels',y.shape)
print('training, before balanced',X_train.shape)
print('labels, before balanced',y_train.shape)

a = 0
b = 0
for j in range(y_train.shape[0]):
    if y_train[j] == -1:
        a += 1
    if y_train[j] == 1:
        b += 1

print('~~~~~')
print('the number of qualified before balanced：',a)
print('the number of unqualified before balanced：',b)
print('~~~~~')

smote = SMOTEENN()
X_train, y_train = smote.fit_sample(X_train,y_train)
print('before permutation:',y_train)
permutation_way2 = np.random.permutation(X_train.shape[0])
X_train = X_train[permutation_way2, :]
y_train = y_train[permutation_way2]
print('after permutation:',y_train)

#verification after permutation
print('training',X_guiyi_result.shape)
print('labels',y.shape)
print('training before balanced',X_train.shape)
print('labels before balanced',y_train.shape)
#test whether the result become larger
a = 0
b = 0
for j in range(y_train.shape[0]):
    if y_train[j] == -1:
        a += 1
    if y_train[j] == 1:
        b += 1

print('~~~~~')
print('the number of qualified after balanced：',a)
print('the number of unqualified after balanced：',b)
print('~~~~~')

#the eighth step: classification
model = AdaBoostClassifier()



#setting the hyper-parameters
GA_population_number = 10
GA_generation = 10
GA_crossover_probability = 0.8
GA_mutate_probability = 0.4

##the ninth step: GA
fsga = FeatureSelectionGA(model, x=X_train, y=y_train,x_test=X_test,y_test=y_test,x_development=X_development,y_development=y_development)
pop = fsga.generate(n_pop=GA_population_number, cxpb=GA_crossover_probability, mutxpb=GA_mutate_probability, ngen=GA_generation)

end_time = time.time()

print(fsga.best_ind)
list_tezheng_selected = []
for i in range(100):
    if fsga.best_ind[i] == 1:
        list_tezheng_selected.append(feature[i])

print(list_tezheng_selected)
print("---lasted %s seconds ---" % str(time.time() - start_time))







