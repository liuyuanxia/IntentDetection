import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(16,5))

#snips
x = [10,25,50,75,100]
#snips
f1_cnn_snips=[94.4,95.2,95.8,96.3,96.9,96.8,96.8,96.6,97.2,97.0]
f1_capsule_snips=[95.6,95.6,96.2,96.3,96.5,96.9,96.8,96.8,97.3,97.7]
acc_cnn_snips=[94.29,95.6,96.86,97.0,97.0]
acc_capsule_snips=[95.57,96,96.43,96.8,97.7]

#UserIntent
f1_cnn_ecdt=[85.7,91.4,92.1,94.7,95.3,93.6,94.5,95.2,95,96]
f1_capsule_ecdt=[88.2,91.8,93.1,94.1,94.4,94.3,95.3,95.6,95.4,96.3]
acc_cnn_ecdt=[88.01,91.7,94.9,95.05,96]
acc_capsule_ecdt=[89.66,92.8,94.3,95.2,96.3]

#FDQ
f1_cnn_fdq=[66.7,71.1,71.9,74.1,77.2,78.5,78.4,78.9,79.9,80.8]
f1_capsule_fdq=[65.9,70.8,70.8,75,74.8,77.6,79.3,80.5,81.9,81.4]
acc_cnn_fdq=[69.02,76.72,79.22,80.43,84.9]
acc_capsule_fdq=[70.32,76.72,78.39,82.47,85.6]



#plt.figure(figsize=(6,6))

plt.subplot(131)
#plt.tight_layout()
#plt.bar(x, f1_cnn_snips,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_snips,width=width, label='F1_BERT_Cap', color='orange')

plt.ylim(94,100)
#plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_snips,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_snips, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_snips,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_snips, marker='h', ms=6,color='r',linestyle='--',label='F1_BERT_Cap')
title='SNIPS'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([10,25,50,75,100])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(132)
#plt.tight_layout()
#plt.bar(x, f1_cnn_ecdt,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_ecdt,width=width, label='F1_BERT_Cap', color='orange')
plt.ylim(85,100)
#plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_ecdt,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_ecdt, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_ecdt,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_ecdt, marker='h', ms=6,color='r',linestyle='--',label='F1_BERT_Cap')
title='ECDT'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([10,25,50,75,100])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(133)
#plt.tight_layout()
#plt.bar(x, f1_cnn_fdq,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_fdq,width=width, label='F1_BERT_Cap', color='orange')
plt.ylim(65,90)
#plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_fdq,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_fdq, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_fdq,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_fdq, marker='h',color='r', linestyle='--',ms=6,label='F1_BERT_Cap')
title='FDQuestion'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([10,25,50,75,100])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.savefig("ACC_f1.svg")
plt.show()
