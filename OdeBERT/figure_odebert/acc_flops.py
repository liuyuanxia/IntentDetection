import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
#plt.figure(dpi=200,figsize=(30,8))

'''
#shoping,layers  1-11
x = [1,2,3,4,5,6,7,8,9,10,11,12]
Dee_3=[0.9700,0.9705,0.9699,0.9688,0.9587,0.9550,0.9491,0.9440,0.9294,0.8862,0.8774,0.8385]
Our_3=[0.9689,0.9688,0.9688,0.969,0.9691,0.9686,0.9669,0.9683,0.9626,0.952,0.9492,0.9249]
LMCL_3=[0.9708,0.9709,0.9705,0.9710,0.9691,0.9688,0.9680,0.9671,0.9641,0.9562,0.9512,0.9317]
#UserIntent

x = [1,2,3,4,5,6,7,8,9,10,11,12]
Dee_1=[0.6342,0.5037,0.1814,0.4723,0.3163,0.1814,0.2054,0.2444,0.3118,0.4588,0.8486,0.9565]
Our_1=[0.7256,0.8966,0.9205,0.9415,0.9490,0.94,0.9415,0.9415,0.9415,0.9415,0.9415,0.9415]
#LMCL_1=[0.8981,0.9385,0.9385,0.9445,0.946,0.952,0.9475,0.9505,0.9535,0.949,0.952,0.9535]
LMCL_1=[0.9595,0.958,0.964,0.962,0.958,0.958,0.9595,0.946,0.9385,0.937,0.9295,0.8906]


#snips
Dee_2=[0.9500,0.97,0.97,0.97,0.9671,0.97,0.9714,0.9729,0.9714,0.9729,0.9757,0.9729]
Our_2=[0.9614,0.9714,0.9729,0.9743,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757]
LMCL_2=[0.9671,0.9757,0.9743,0.9771,0.9771,0.9757,0.9757,0.9757,0.9743,0.9757,0.9757,0.9757]
#chnsenticorp

Dee_4=[0.7525,0.795,0.7992,0.8525,0.8775,0.8908,0.8958,0.8992,0.9167,0.9283,0.9417,0.9517]
Our_4=[0.8542,0.8942,0.8942,0.915,0.935,0.9442,0.9475,0.9483,0.9508,0.9483,0.9483,0.9500]
LMCL_4=[0.9033,0.9242,0.9275,0.9458,0.9542,0.9525,0.9592,0.9583,0.9542,0.9542,0.9550,0.9542]


#FDQ
Dee_5=[0.5399,0.5121,0.4462,0.5297,0.4805,0.4527,0.4629,0.4703,0.6011,0.6512,0.8126,0.8451]
Our_5=[0.5993,0.6957,0.7607,0.7968,0.8052,0.8200,0.8275,0.8377,0.8404,0.8432,0.8460,0.8414]
LMCL_5=[0.7171,0.7347,0.769,0.8089,0.8293,0.834,0.8321,0.834,0.8525,0.8488,0.8442,0.8534]

plt.subplot(132)
#plt.tight_layout()
plt.plot(x, Dee_1,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_1, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='ECDT(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
plt.legend(fontsize=15)

plt.subplot(131)
#plt.tight_layout()
plt.plot(x, Dee_2,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_2, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_2, marker='v', ms=5,label='OdeBERT')
title='SNIPS(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
plt.legend(fontsize=15)

plt.subplot(133)
#plt.tight_layout()
plt.plot(x, Dee_5,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_5, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_5, marker='v', ms=5,label='OdeBERT')
title='FDQuestion(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
plt.legend(fontsize=8)

plt.subplot(224)
#plt.tight_layout()
plt.plot(x, Dee_4,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_4, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_4, marker='v', ms=5,label='OdeBERT')
title='Chnsenticorp(Dynamic Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
plt.legend(fontsize=8)

'''
x = [1,1.090,1.199,1.332,1.499,1.713,1.998,2.398,2.998,3.997,5.996,11.99]
#SNIPS
Dee_1=[0.9729,0.9757,0.9729,0.9714,0.9729,0.9714,0.9700,0.9671,0.97,0.9700,0.97,0.9500]
Our_1=[0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9743,0.9729,0.9714,0.9614]
#cnn_2=[0.9714,0.9714,0.97,0.9714,0.97,0.97,0.9729,0.9729,0.9729,0.9729,0.9743,0.9714]
LMCL_1=[0.9671,0.9757,0.9743,0.9771,0.9771,0.9757,0.9757,0.9757,0.9743,0.9757,0.9757,0.9757]
#UserIntent
Dee_2=[0.9565,0.8486,0.4588,0.3118,0.2444,0.2054,0.1814,0.3163,0.4723,0.1814,0.5037,0.6342]
Our_2=[0.9415,0.9415,0.9415,0.9415,0.9415,0.9415,0.94,0.9490,0.9415,0.9205,0.8966,0.7256]
LMCL_ECDT=[0.9595,0.958,0.964,0.962,0.958,0.958,0.9595,0.946,0.9385,0.937,0.9295,0.8906]
#cnn_1=[0.94,0.94,0.94,0.9415,0.9415,0.943,0.946,0.946,0.949,0.9415,0.94,0.925]
#LMCL_1=[0.8981,0.9385,0.9385,0.9445,0.946,0.952,0.9475,0.9505,0.9535,0.949,0.952,0.9535]
LMCL_2=LMCL_ECDT

#FDQusetion
Dee_3=[0.5399,0.5121,0.4462,0.5297,0.4805,0.4527,0.4629,0.4703,0.6011,0.6512,0.8126,0.8451]
Our_3=[0.5993,0.6957,0.7607,0.7968,0.8052,0.8200,0.8275,0.8377,0.8404,0.8432,0.8460,0.8414]
LMCL_3=[0.7171,0.7347,0.769,0.8089,0.8293,0.834,0.8321,0.834,0.8525,0.8488,0.8442,0.8534]

#shopping  acc+FLOPs
Dee_4=[0.9700 ,0.9705,0.9699,0.9688,0.9587,0.9550,0.9491,0.9440,0.9294,0.8862,0.8774,0.8385]
Our_4=[0.9689,0.9688,0.9688,0.969,0.9691,0.9686,0.9669,0.9683,0.9626,0.952,0.9492,0.9249]
#x2=[0.802,0.875,0.963,1.070,1.204,1.376,1.605,1.926,2.408,3.211,4.816,9.633]
LMCL_4=[0.9708,0.9709,0.9705,0.9710,0.9691,0.9688,0.9680,0.9671,0.9641,0.9562,0.9512,0.9317]

#chnsenticorp
Dee_5=[0.9517,0.9417,0.9283,0.9167,0.8992,0.8958,0.8908,0.8775,0.8525,0.7992,0.795,0.7525]
Our_5=[0.95,0.9483,0.9483,0.9508,0.9483,0.9475,0.9442,0.935,0.915,0.8942,0.8942,0.8542]
#LMCL_4=[0.9508,0.9508,0.9483,0.95,0.9492,0.9492,0.9467,0.9450,0.9333,0.9117,0.9067,0.8867]
LMCL_5=[0.9033,0.9242,0.9275,0.9458,0.9542,0.9525,0.9592,0.9583,0.9542,0.9542,0.9550,0.9542]


plt.subplot(151)
#plt.tight_layout()
plt.plot(x, Dee_1,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_1, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='SNIPS(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Spe')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=12)

plt.subplot(152)
#plt.tight_layout()
plt.plot(x, Dee_2,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_2, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_2, marker='v', ms=5,label='OdeBERT')
title='ECDT(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Spe')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=12)

plt.subplot(153)
#plt.tight_layout()
plt.plot(x, Dee_3[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_3[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_3[::-1], marker='v', ms=5,label='OdeBERT')
title='FDQestion(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Spe')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=8)

plt.subplot(154)
#plt.tight_layout()
plt.plot(x, Dee_4,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_4, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_4, marker='v', ms=5,label='OdeBERT')
title='Shopping(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Spe')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=8)

plt.subplot(155)
#plt.tight_layout()
plt.plot(x, Dee_5,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, Our_5, marker='h', ms=5,label='OeBERT')
plt.plot(x, LMCL_5[::-1], marker='v', ms=5,label='OdeBERT')
title='Chnsenticorp(Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Spe')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=8)

'''
Dee_ACC_1=[0.9565,0.9535,0.943,0.937,0.9175,0.8771,0.8186,0.7136,0.6387]
Our_ACC_1=[0.9385,0.94,0.9295,0.91,0.8906,0.8606,0.8336,0.7706,0.7316]
cnn_ACC_1=[0.9415,0.9445,0.9385,0.934,0.934,0.934,0.928,0.925,0.925]
Dee_FLOPS_1=[1.117,1.173,1.231,1.326,1.467,1.743,2.423,4.607,11.04]
Our_FLOPS_1=[3.446,4.412,5.329,6.205,7.174,8.289,9.366,10.98,11.86]
cnn_FLOPS_1=[3.650,4.889,6.148,7.195,8.102,8.729,9.049,9.421,9.632]
#SNIPS
Dee_ACC_2=[0.9729,0.9714,0.9657,0.9586,0.9557,0.9529,0.9514,0.9500,0.9500]
Our_ACC_2=[0.9757,0.9729,0.97,0.9629,0.9614,0.9614,0.96,0.9614,0.9614]
cnn_ACC_2=[0.9714,0.9714,0.9714,0.9714,0.9729,0.9729,0.9714,0.9714,0.9714]
Dee_FLOPS_2=[8.262,9.317,10.46,11.35,11.57,11.82,11.94,11.99,11.99]
Our_FLOPS_2=[8.575,9.660,10.88,11.42,11.75,11.82,11.94,11.99,11.99]
cnn_FLOPS_2=[6.756,7.697,8.428,9.137,9.444,9.524,9.591,9.605,9.619]
#shopping
Dee_ACC_3=[0.9703,0.9703,0.9688,0.968,0.9626,0.9557,0.9446,0.9292,0.9046]
Our_ACC_3=[0.9688,0.9669,0.965,0.9628,0.9593,0.956,0.9532,0.9491,0.943]
cnn_ACC_3=[0.9685,0.9688,0.9687,0.9684,0.9673,0.9655,0.9646,0.9628,0.9590]
Dee_FLOPS_3=[1.725,2.078,2.424,2.821,3.308,3.933,4.744,5.886,7.687]
Our_FLOPS_3=[4.419,5.507,6.397,7.179,7.857,8.562,9.245,9.938,10.74]
cnn_FLOPS_3=[2.847,3.644,4.340,4.974,5.607,6.238,6.865,7.619,8.456]
#chnsenticorp
Dee_ACC_4=[0.9517,0.9508,0.9492,0.9467,0.9400,0.9325,0.925,0.8983,0.8592]
Our_ACC_4=[0.9517,0.9483,0.9475,0.9425,0.935,0.9292,0.9208,0.91,0.8975]
cnn_ACC_4=[0.9508,0.9571,0.9508,0.9483,0.9458,0.9417,0.9317,0.9258,0.915]
Dee_FLOPS_4=[1.154,1.274,1.391,1.535,1.733,2.030,2.404,3.142,4.834]
Our_FLOPS_4=[2.264,2.936,3.619,4.270,4.971,5.826,6.827,7.885,9.266]
cnn_FLOPS_4=[1.856,2.378,2.857,3.366,3.926,4.579,5.278,6.279,7.358]

#plt.figure(figsize=(8,8))
plt.subplot(221)
plt.tight_layout()
plt.plot(Dee_FLOPS_1, Dee_ACC_1,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(Our_FLOPS_1, Our_ACC_1, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_1, marker='v', ms=5,label='Our+cnn')
title='ECDT(S)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(222)
plt.tight_layout()
plt.plot(Dee_FLOPS_2, Dee_ACC_2,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(Our_FLOPS_2, Our_ACC_2, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_2[::-1], marker='v', ms=5,label='Our+cnn')
title='SNIPS(S)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(223)
plt.tight_layout()
plt.plot(Dee_FLOPS_3, Dee_ACC_3,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(Our_FLOPS_3, Our_ACC_3, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_3, marker='v', ms=5,label='Our+cnn')
title='shopping(S)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(224)
plt.tight_layout()
plt.plot(Dee_FLOPS_4, Dee_ACC_4,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(Our_FLOPS_4, Our_ACC_4, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_4, marker='v', ms=5,label='Our+cnn')
title='chnsenticorp(S)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)


#plt.figure(figsize=(8,8))
plt.subplot(221)
plt.tight_layout()
plt.plot(x1, Dee_1,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_1, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_1, marker='v', ms=5,label='Our+cnn')
title='ECDT(Layers)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(222)
plt.tight_layout()
plt.plot(x1, Dee_2,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_2, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_2[::-1], marker='v', ms=5,label='Our+cnn')
title='SNIPS(Layers)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(223)
plt.tight_layout()
plt.plot(x1, Dee_3,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_3, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_3, marker='v', ms=5,label='Our+cnn')
title='shopping(Layers)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

plt.subplot(224)
plt.tight_layout()
plt.plot(x1, Dee_4,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_4, marker='h', ms=5,label='Our')
#plt.plot(x2, cnn_4, marker='v', ms=5,label='Our+cnn')
title='chnsenticorp(Layers)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('speedup(X)')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Accuracy')
plt.legend(fontsize=7)

#shoping,layers  1-11
x1 = [1,2,3,4,5,6,7,8,9,10,11,12]
x2= [1,1.090,1.199,1.332,1.499,1.713,1.998,2.398,2.998,3.997,5.996,11.99]
Dee_3=[0.9700,0.9705,0.9699,0.9688,0.9587,0.9550,0.9491,0.9440,0.9294,0.8862,0.8774,0.8385]
Our_3=[0.9689,0.9688,0.9688,0.969,0.9691,0.9686,0.9669,0.9683,0.9626,0.952,0.9492,0.9249]
LMCL_3=[0.9708,0.9709,0.9705,0.9710,0.9691,0.9688,0.9680,0.9671,0.9641,0.9562,0.9512,0.9317]
#UserIntent


Dee_1=[0.6342,0.5037,0.1814,0.4723,0.3163,0.1814,0.2054,0.2444,0.3118,0.4588,0.8486,0.9565]
Our_1=[0.7256,0.8966,0.9205,0.9415,0.9490,0.94,0.9415,0.9415,0.9415,0.9415,0.9415,0.9415]
#LMCL_1=[0.8981,0.9385,0.9385,0.9445,0.946,0.952,0.9475,0.9505,0.9535,0.949,0.952,0.9535]
LMCL_1=[0.9595,0.958,0.964,0.962,0.958,0.958,0.9595,0.946,0.9385,0.937,0.9295,0.8906]


#snips
Dee_2=[0.9500,0.97,0.97,0.97,0.9671,0.97,0.9714,0.9729,0.9714,0.9729,0.9757,0.9729]
Our_2=[0.9614,0.9714,0.9729,0.9743,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757,0.9757]
LMCL_2=[0.9671,0.9757,0.9743,0.9771,0.9771,0.9757,0.9757,0.9757,0.9743,0.9757,0.9757,0.9757]
#chnsenticorp

Dee_4=[0.7525,0.795,0.7992,0.8525,0.8775,0.8908,0.8958,0.8992,0.9167,0.9283,0.9417,0.9517]
Our_4=[0.8542,0.8942,0.8942,0.915,0.935,0.9442,0.9475,0.9483,0.9508,0.9483,0.9483,0.9500]
LMCL_4=[0.9033,0.9242,0.9275,0.9458,0.9542,0.9525,0.9592,0.9583,0.9542,0.9542,0.9550,0.9542]


#FDQ
Dee_5=[0.5399,0.5121,0.4462,0.5297,0.4805,0.4527,0.4629,0.4703,0.6011,0.6512,0.8126,0.8451]
Our_5=[0.5993,0.6957,0.7607,0.7968,0.8052,0.8200,0.8275,0.8377,0.8404,0.8432,0.8460,0.8414]
LMCL_5=[0.6957,0.7384,0.7616,0.7922,0.8247,0.8423,0.8404,0.8442,0.8460,0.8479,0.8506,0.8488]

plt.subplot(521)
plt.tight_layout()
plt.plot(x1, Dee_1,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_1, marker='h', ms=5,label='OeBERT')
plt.plot(x1, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='ECDT(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(523)
plt.tight_layout()
plt.plot(x1, Dee_2,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_2, marker='h', ms=5,label='OeBERT')
plt.plot(x1, LMCL_2, marker='v', ms=5,label='OdeBERT')
title='SNIPS(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(525)
plt.tight_layout()
plt.plot(x1, Dee_3[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_3[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x1, LMCL_3[::-1], marker='v', ms=5,label='OdeBERT')
title='shopping(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(527)
plt.tight_layout()
plt.plot(x1, Dee_4,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_4, marker='h', ms=5,label='OeBERT')
plt.plot(x1, LMCL_4, marker='v', ms=5,label='OdeBERT')
title='chnsenticorp(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(529)
plt.tight_layout()
plt.plot(x1, Dee_5,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x1, Our_5, marker='h', ms=5,label='OeBERT')
plt.plot(x1, LMCL_5, marker='v', ms=5,label='OdeBERT')
title='FDQuestion(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(522)
plt.tight_layout()
plt.plot(x2, Dee_1[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x2, Our_1[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x2, LMCL_1, marker='v', ms=5,label='OdeBERT')
title='ECDT(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speedup')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(524)
plt.tight_layout()
plt.plot(x2, Dee_2[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x2, Our_2[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x2, LMCL_2, marker='v', ms=5,label='OdeBERT')
title='SNIPS(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speedup')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(526)
plt.tight_layout()
plt.plot(x2, Dee_3[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x2, Our_3[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x2, LMCL_3[::-1], marker='v', ms=5,label='OdeBERT')
title='shopping(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speedup')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(528)
plt.tight_layout()
plt.plot(x2, Dee_4[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x2, Our_4[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x2, LMCL_4, marker='v', ms=5,label='OdeBERT')
title='chnsenticorp(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speedup')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)

plt.subplot(5,2,10)
plt.tight_layout()
plt.plot(x2, Dee_5[::-1],marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x2, Our_5[::-1], marker='h', ms=5,label='OeBERT')
plt.plot(x2, LMCL_5[::-1], marker='v', ms=5,label='OdeBERT')
title='FDQuestion(Scalable Depth)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Speedup')
#plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Accuracy')
#plt.legend(fontsize=8)
'''


#plt.savefig("./ACC+FLOPs.png")
plt.show()


