import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(16,5))

#snips
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#snips
f1_cnn_snips=[0.944,0.952,0.958,0.963,0.969,0.968,0.968,0.966,0.972,0.970]
f1_capsule_snips=[0.956,0.956,0.962,0.963,0.965,0.969,0.968,0.968,0.973,0.977]
acc_cnn_snips=[0.9429,0.9514,0.9571,0.9629,0.9686,0.9671,0.9671,0.9657,0.9717,0.97]
acc_capsule_snips=[0.9557,0.9557,0.9614,0.9629,0.9643,0.9686,0.9671,0.9671,0.9729,0.9770]

#UserIntent
f1_cnn_ecdt=[0.857,0.914,0.921,0.947,0.953,0.936,0.945,0.952,0.95,0.96]
f1_capsule_ecdt=[0.882,0.918,0.931,0.941,0.944,0.943,0.953,0.956,0.954,0.963]
acc_cnn_ecdt=[0.8801,0.919,0.922,0.9445,0.949,0.934,0.94,0.9505,0.949,0.96]
acc_capsule_ecdt=[0.8966,0.9205,0.9326,0.94,0.943,0.9445,0.9505,0.9535,0.955,0.963]

#FDQ
f1_cnn_fdq=[0.667,0.711,0.719,0.741,0.772,0.785,0.784,0.789,0.799,0.808]
f1_capsule_fdq=[0.659,0.708,0.708,0.75,0.748,0.776,0.793,0.805,0.819,0.814]
acc_cnn_fdq=[0.6902,0.7468,0.7681,0.7746,0.7922,0.8145,0.8135,0.8237,0.8321,0.849]
acc_capsule_fdq=[0.7032,0.7551,0.7468,0.7885,0.7839,0.8173,0.821,0.8349,0.8423,0.856]



#plt.figure(figsize=(6,6))

plt.subplot(131)
#plt.tight_layout()
#plt.bar(x, f1_cnn_snips,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_snips,width=width, label='F1_BERT_Cap', color='orange')

plt.ylim(0.94,1.0)
plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_snips,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_snips, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_snips,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_snips, marker='h', ms=6,color='r',linestyle='--',label='F1_BERT_Cap')
title='SNIPS'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(132)
#plt.tight_layout()
#plt.bar(x, f1_cnn_ecdt,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_ecdt,width=width, label='F1_BERT_Cap', color='orange')
plt.ylim(0.85,1.0)
plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_ecdt,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_ecdt, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_ecdt,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_ecdt, marker='h', ms=6,color='r',linestyle='--',label='F1_BERT_Cap')
title='ECDT'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(133)
#plt.tight_layout()
#plt.bar(x, f1_cnn_fdq,  width=width, label='F1_RBERT-C',color='#427DA6')
#plt.bar(x + width, f1_capsule_fdq,width=width, label='F1_BERT_Cap', color='orange')
plt.ylim(0.65,0.9)
plt.xlim(0.1,1.0)
#plt.tight_layout()
#plt.scatter(,0.9757,label='BERT')
plt.plot(x, acc_cnn_fdq,marker='h', ms=6,color='b',label='Acc_RBERT-C')
plt.plot(x, acc_capsule_fdq, marker='h', ms=6,color='r',label='Acc_BERT_Cap')
#plt.plot(x, f1_cnn_fdq,marker='h', ms=6,color='b',linestyle='--',label='F1_RBERT-C')
#plt.plot(x, f1_capsule_fdq, marker='h',color='r', linestyle='--',ms=6,label='F1_BERT_Cap')
title='FDQuestion'
plt.title(title)
plt.xlabel('Proportion of training data')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.ylabel('Performance')
plt.legend(fontsize=7)
plt.grid(ls='--')
'''
#SNIPS
x=[1,1.090,1.199,1.332,1.499,1.713,1.998,2.398,2.998,3.997,5.996,11.99]
Dee_1_flops=[0.9757,0.9729,0.9743,0.9743,0.9743,0.9743,0.9743,0.9671,0.97,0.9700,0.9714,0.9471]
Fast_1_flops=[0.9471,0.9714,0.9686,0.97,0.9671,0.9743,0.9729,0.9743,0.9743,0.9743,0.9743,0.9757][::-1]
RightTool_1_flops=[0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9743,0.9729,0.9714,0.9614]
RomeBERT_1_flops=[0.9643,0.9714,0.9700,0.9743,0.9743,0.9743,0.9743,0.9729,0.9729,0.9729,0.9729,0.9729][::-1]
#cnn_2=[0.9714,0.9714,0.97,0.9714,0.97,0.97,0.9729,0.9729,0.9729,0.9729,0.9743,0.9714]
LMCL_1_flops=[0.9671,0.9757,0.9743,0.9771,0.9771,0.9757,0.9757,0.9757,0.9743,0.9757,0.9757,0.9757][::-1]

#UserIntent
Dee_2_flops=[0.6177,0.5247,0.2759,0.5307,0.3433,0.1754,0.2069,0.2084,0.2474,0.4138,0.8486,0.9565][::-1]
Fast_2_flops=[0.6237,0.5232,0.2804,0.5307,0.3493,0.1814,0.2069,0.2084,0.2519,0.4153,0.8456,0.9565][::-1]
RightTool_2_flops=[0.7496,0.8891,0.9280,0.9370,0.9460,0.9430,0.9415,0.9415,0.9400,0.9415,0.9445,0.9445][::-1]
RomeBERT_2_flops=[0.8396,0.904,0.9235,0.9235,0.9355,0.937,0.9415,0.9415,0.9415,0.9400,0.9400,0.943][::-1]
LMCL_2_flops=[0.9595,0.958,0.964,0.9625,0.958,0.958,0.9595,0.946,0.937,0.937,0.9295,0.8891]

#FDQ
Dee_3_flops=[0.6243,0.6048,0.5519,0.6865,0.6438,0.6076,0.6456,0.6920,0.7699,0.8061,0.8349,0.8451][::-1]
Fast_3_flops=[0.6215,0.5983,0.5501,0.692,0.6521,0.6095,0.6586,0.6939,0.7737,0.8089,0.8386,0.8451][::-1]
RightTool_3_flops=[0.5993,0.6957,0.7607,0.7968,0.8052,0.8200,0.8275,0.8377,0.8404,0.8432,0.8460,0.8414][::-1]
RomeBERT_3_flops=[0.7282,0.7829,0.7839,0.8153,0.8237,0.8210,0.8191,0.8153,0.8145,0.8145,0.8135,0.8107][::-1]
LMCL_3_flops=[0.7171,0.7347,0.769,0.8089,0.8293,0.834,0.8321,0.834,0.8525,0.8488,0.8442,0.8534][::-1]

plt.subplot(234)
#plt.tight_layout()
plt.scatter(1.0,0.9757,label='BERT')
plt.plot(x, Dee_1_flops,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_1_flops,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_1_flops, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_1_flops,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_1_flops, marker='v', ms=6,label='OdeBERT')

title='SNIPS(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speed-up')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(235)
#plt.tight_layout()
plt.scatter(1.0,0.9565,label='BERT')
plt.plot(x, Dee_2_flops,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_2_flops,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_2_flops, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_2_flops,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_2_flops, marker='v', ms=6,label='OdeBERT')
title='ECDT(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speed-up')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
#plt.xticks()
plt.ylabel('Acc')
plt.legend(fontsize=7)
plt.grid(ls='--')

plt.subplot(236)
#plt.tight_layout()
plt.scatter(1.0,0.8451,label='BERT')
plt.plot(x, Dee_3_flops,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_3_flops,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_3_flops, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_3_flops,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_3_flops, marker='v', ms=6,label='OdeBERT')
title='FDQuestion(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speed-up')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
#plt.xticks([1,1.090,1.199,1.332,1.499,1.713,1.998,2.398,2.998,3.997,5.996,11.99])
plt.ylabel('Acc')
plt.legend(fontsize=7)
plt.grid(ls='--')
#plt.savefig("./ACC+FLOPs.png")
'''
plt.savefig("ACC_f1.svg")
plt.show()
