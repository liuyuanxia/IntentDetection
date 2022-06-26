import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(16,5))
font={'family':'Times New Roman',
      'weight':'normal','size':12,
}

x = [1,2,3,4,5,6,7,8,9,10,11,12]
#snips
Dee_1=[97.57,97.29,97.43,97.43,97.43,97.43,97.43,96.71,97,97.00,97.14,94.71][::-1]
Fast_1=[94.71,97.14,96.86,97,96.71,97.43,97.29,97.43,97.43,97.43,97.43,97.57]
RightTool_1=[96.14,97.14,97.29,97.43,97.57,97.57,97.57,97.57,97.57,97.57,97.57,97.57]
RomeBERT_1=[96.43,97.14,97.00,97.43,97.43,97.43,97.43,97.29,97.29,97.29,97.29,97.29]
LMCL_1=[96.71,97.57,97.43,97.71,97.71,97.57,97.57,97.57,97.43,97.57,97.57,97.57]

#UserIntent
Dee_2=[61.77,52.47,27.59,53.07,34.33,17.54,20.69,20.84,24.74,41.38,84.86,95.65]
Fast_2=[62.37,52.32,28.04,53.07,34.93,18.14,20.69,20.84,25.19,41.53,84.56,95.65]
RightTool_2=[74.96,88.91,92.80,93.70,94.60,94.30,94.15,94.15,94.00,94.15,94.45,94.45]
RomeBERT_2=[83.96,90.4,92.35,92.35,93.55,93.7,94.15,94.15,94.15,94.00,94.00,94.3]
LMCL_2=[95.95,95.8,96.4,96.25,95.8,95.8,95.95,94.6,93.7,93.7,92.95,88.91][::-1]

#FDQ
Dee_3=[62.43,60.48,55.19,68.65,64.38,60.76,64.56,69.20,76.99,80.61,83.49,84.51]
Fast_3=[62.15,59.83,55.01,69.2,65.21,60.95,65.86,69.39,77.37,80.89,83.86,84.51]
RightTool_3=[59.93,69.57,76.07,79.68,80.52,82.00,82.75,83.77,84.04,84.32,84.60,84.14]
RomeBERT_3=[76.9,78.48,80.24,80.61,80.89,81.17,81.08,80.98,80.98,81.08,80.80,80.98]
LMCL_3=[71.71,73.47,76.9,80.89,82.93,83.4,83.21,83.4,85.25,84.88,84.42,85.34]

print('hashadghhsfdhs')
plt.subplot(131)
#plt.tight_layout()
plt.scatter(12,0.9757,label='BERT')
plt.plot(x, Dee_1,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_1,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_1, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_1,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_1, marker='v', ms=6,label='OdeBERT')
title='SNIPS(Depth)'
plt.title(title)
plt.ylim(92,98)
#plt.title('line chart')
plt.xlabel('Layers',fontdict=font)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc',fontdict=font)
plt.legend(fontsize=7,loc=4)
plt.grid(ls='--')

plt.subplot(132)
#plt.tight_layout()
plt.scatter(12,0.9565,label='BERT')
plt.plot(x, Dee_2,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_2,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_2, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_2,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_2, marker='v', ms=6,label='OdeBERT')
title='ECDT(Depth)'
plt.title(title)
plt.ylim(10,100)
#plt.title('line chart')
plt.xlabel('Layers',fontdict={'family' : 'Times New Roman',})
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
fontdict = {'family':'Times New Roman',}
plt.legend(fontsize=7,loc=4,prop=fontdict)
plt.grid(ls='--')

plt.subplot(133)
#plt.tight_layout()
plt.scatter(12,0.8451,label='BERT')
plt.plot(x, Dee_3,marker='x', ms=6,label='DeeBERT')
#plt.plot(x, Fast_3,marker='>', ms=6,label='FastBERT')
plt.plot(x, RightTool_3, marker='h', ms=6,label='RightTool')
plt.plot(x, RomeBERT_3,marker='p', ms=6,label='RomeBERT')
plt.plot(x, LMCL_3, marker='v', ms=6,label='OdeBERT')
title='FDQuestion(Depth)'
plt.title(title)
plt.ylim(40,90)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=7,loc=4)
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
plt.savefig("ACC_layer.svg")
plt.show()
