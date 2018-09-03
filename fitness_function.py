#this is the fitness function calculation
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve

class FitenessFunction:
    def __init__(self, n_splits=3, *args, **kwargs):

        self.n_splits = n_splits

        self.f1 = []
    def calculate_fitness(self, model, x, y, x_test, y_test, x_development, y_development, ):
        model.fit(x, y)
        predicted_y_train = model.predict(x)
        predicted_y_test = model.predict(x_test)
        predicted_y_development = model.predict(x_development)

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
        print(classification_report(y_test, predicted_y_test))

        self.f1.append(f1_score_test)

        return recall_score_test
    def get_f1(self):
        return self.f1
class keshihua():
    def huatu(self, model, x, y, x_test, y_test, x_development, y_development,):
        model.fit(x, y)
        predicted_y_train = model.predict(x)
        predicted_y_test = model.predict(x_test)
        predicted_y_development = model.predict(x_development)

        precisions, recalls, thresholds = precision_recall_curve(y_test, predicted_y_test)
        ax = pl.subplot(2, 1, 2)
        ax.plot(recalls, precisions)
        ax.set_ylim([0.0, 1.0])
        ax.set_title('Precision recall curve')
        plt.xlabel("recall")
        plt.ylabel("precision")
        pl.show()

        PRE_Y=model.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, PRE_Y[:,1], pos_label=1)
        plt.plot(fpr, tpr, linewidth=2, label="ROC")
        plt.xlabel("false presitive rate")

        plt.ylabel("true presitive rate")

        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.legend(loc=4)
        plt.show()
