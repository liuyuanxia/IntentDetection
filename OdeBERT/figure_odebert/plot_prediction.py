import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='confusion Matrix', cmap=plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
        

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        num='{:.3f}'.format(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > str(thresh) else "black")
    
    plt.ylabel('Prediction')
    plt.xlabel('True Label')
    
    plt.tight_layout()
    plt.savefig('method_2.png', transparent=True, dpi=800) 
    
    plt.show()

model1="/home/yuanxia/yanyixia/KnowledgeDistillation/FastBERT-master/datasets/UserIntent_label/attention_cnn.txt"
attention=np.loadtxt(model1)

"""method 2"""
if True:
    label = ['1','2','3','4','5','6','7']
    plot_confusion_matrix(attention, label)
