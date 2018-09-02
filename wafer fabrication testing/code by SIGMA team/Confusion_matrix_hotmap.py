# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class confusionMatrixHotMap:
    def __init__(self, labels, y_true, y_pred):
        self.labels = labels
        self.y_true = y_true
        self.y_pred = y_pred

    def show_confusion_matrix(self):
        tick_marks = np.array(range(len(self.labels))) + 0.5
        cm = confusion_matrix(self.y_true, self.y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print (cm_normalized)
        plt.figure(figsize=(6, 4), dpi=120)

        ind_array = np.arange(len(self.labels))
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='orange', fontsize=10, va='center', ha='center')
        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        self.plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        plt.savefig('confusion_matrix.png', format='png')
        plt.show()

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(self.labels)))
        plt.xticks(xlocations, self.labels, rotation=90)
        plt.yticks(xlocations, self.labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')