#coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
my_font=font_manager.FontProperties(fname='./SIMSUN.TTC')
#plt.rc('font',family='YouYuan')

#plt.figure(figsize=(12,3))
recallList=[82.9,90.5,83.3,85.2,88.9,67.6,84.6,85.1,87.5]
labels=["事实类","原因类","推荐类","描述类","方法类","是非类","枚举类","评价类","需求类"]
width=0.3
index=np.arange(len(labels))
plt.bar(index,recallList,width,color='coral')

plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0],fontsize=12)
plt.ylabel('Recall')
for a,b in zip(index,recallList):
    plt.text(a,b,b,ha='center',va='bottom')
plt.show()


