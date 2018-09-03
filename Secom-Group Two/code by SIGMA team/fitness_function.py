from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.metrics import confusion_matrix
from Confusion_matrix_hotmap import confusionMatrixHotMap
class FitenessFunction:
    def __init__(self):
        self.predict=[]
        self.true=[]
    def calculate_fitness(self, model, x, y, x_test, y_test, x_development, y_development, ):
        model.fit(x, y)
        predicted_y_train = model.predict(x)
        predicted_y_test = model.predict(x_test)
        predicted_y_development = model.predict(x_development)
        self.predict.append(predicted_y_test)
        f1_score_train = f1_score(y, predicted_y_train)
        f1_score_test = f1_score(y_test, predicted_y_test)
        f1_score_development = f1_score(y_development, predicted_y_development)

        precision_score_train = precision_score(y, predicted_y_train)
        precision_score_test = precision_score(y_test, predicted_y_test)
        precision_score_development = precision_score(y_development, predicted_y_development)

        recall_score_train = recall_score(y, predicted_y_train)
        recall_score_test = recall_score(y_test, predicted_y_test)
        recall_score_development = recall_score(y_development, predicted_y_development)

        train_acc = model.score(x, y)
        test_acc = model.score(x_test, y_test)
        develop_acc = model.score(x_development, y_development)
        #print(predicted_y_test)
        print(classification_report(y_test, predicted_y_test))
        print('Classification report of validation set：')
        print(classification_report(y_development,predicted_y_development))
        return recall_score_test

class keshihua():
    def huatu(self, model, x, y, x_test, y_test, x_development, y_development, ):
        model.fit(x, y)
        predicted_y_train = model.predict(x)
        predicted_y_test = model.predict(x_test)
        predicted_y_development = model.predict(x_development)
        print(classification_report(y_test, predicted_y_test))
        pre_y=model.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, pre_y[:,1], pos_label=1)
        precisions, recalls, thresholds = precision_recall_curve(y_test, predicted_y_test)
        labels = [-1, 1]
        cfm = confusionMatrixHotMap(labels, y_test, predicted_y_test)
        cfm.show_confusion_matrix()
        plt.plot(fpr, tpr, linewidth=2, label="ROC")
        plt.xlabel("false presitive rate")
        plt.ylabel("true presitive rate")
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.legend(loc=4)
        plt.show()
        precisions, recalls, thresholds = precision_recall_curve(y_test, predicted_y_test)
        print(precisions, recalls, thresholds)
        def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
            plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
            plt.plot(thresholds, recalls[:-1], 'r--', label='Recall')
            plt.xlabel("Threshold")
            plt.legend(loc='upper left')
            plt.ylim([0, 1])
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        plt.show()
        ax = pl.subplot(2, 1, 2)
        ax.plot(recalls, precisions)
        ax.set_ylim([0.0, 1.0])
        ax.set_title('Precision recall curve')
        plt.xlabel("recall")
        plt.ylabel("precision")
        pl.show()




