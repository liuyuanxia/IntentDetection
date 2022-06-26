import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(16,5))

#x = np.arange(1,12,dtype=np.int32)
'''
#0.1
Dee_1 = [57,1,0,4,1,0,0,0,0,42,84]
Our_1 = [121,167,124,108,59,21,17,10,8,5,4]
cnn_1 = [334,118,53,64,33,12,9,5,6,7,1]
#0.3
Dee_3 = [120,1,0,0,2,0,0,0,1,32,97]
Our_3 = [210,263,113,41,19,9,3,2,1,1,3]
cnn_3 = [488,99,37,20,11,1,2,3,0,1,1]
#0.5
Dee_5 = [206,6,0,3,3,0,0,0,5,23,123]
Our_5 = [331,256,60,12,6,1,0,1,0,0,0]
cnn_5 = [594,45,16,8,1,2,0,0,0,0,0]
#0.8

Dee_8 = [555,5,0,1,5,0,2,0,1,6,45]
Our_8 = [609,55,3,0,0,0,0,0,0,0,0]
cnn_8 = [655,9,3,0,0,0,0,0,0,0,0]

total_width, n = 0.9, 3
width = total_width / n
x = x - (total_width - width) / 2
plt.figure(figsize=(12,3))

plt.subplot(131)
plt.tight_layout()
plt.tight_layout()
plt.bar(x, Dee_1,  width=width, label='DeeBERT')
plt.bar(x + width, Our_1, width=width, label='Our')
#plt.bar(x + 2 * width, cnn_1, width=width, label='Our+cnn')

title='ECDT(S=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Number of Easy Samples')
plt.legend(fontsize=7)

plt.subplot(132)
plt.tight_layout()
plt.tight_layout()
plt.bar(x, Dee_5,  width=width, label='DeeBERT')
plt.bar(x + width, Our_5, width=width, label='Our')
#plt.bar(x + 2 * width, cnn_5, width=width, label='Our+cnn')

title='ECDT(S=0.5)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Number of Easy Samples')
plt.legend(fontsize=7)

plt.subplot(133)
plt.tight_layout()
plt.tight_layout()
plt.bar(x, Dee_8,  width=width, label='DeeBERT')
plt.bar(x + width, Our_8, width=width, label='Our')
#plt.bar(x + 2 * width, cnn_8, width=width, label='Our+cnn')

title='ECDT(S=0.8)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Number of Easy Samples')
plt.legend(fontsize=7)


'''
#snips
x = np.arange(1,12)
Dee_1=[632,48,11,4,2,2,0,0,1,0,0]
RightTool_1=[673,20,2,3,0,0,0,0,0,0,0]
Rome_1=[684,6,2,0,0,0,1,0,0,0,0]
Ode_1=[695,1,2,0,0,0,1,0,0,0,0]

#UserIntent
Dee_2=[155,19,0,3,0,0,0,0,0,7,140]
RightTool_2=[185,227,105,64,42,10,11,5,6,2,1]
Rome_2=[285,80,18,12,10,10,5,2,1,3,2]
Ode_2=[406,122,41,39,12,10,4,9,8,1,3]

#FDQ
Dee_3=[257,82,3,136,51,9,22,54,126,107,173]
RightTool_3=[203,284,202,179,85,35,20,15,11,9,2]
Rome_3=[597,166,62,18,7,4,1,2,0,1,0]
Ode_3=[223,467,101,113,68,30,6,3,9,3,7]
'''
#Shopping
Dee_4=[189,132,138,3160,1095,52,128,1421,1563,523,520]
Our_4=[4742,2321,715,637,437,241,123,88,123,56,5]
cnn_4=[5092,1252,753,702,423,145,105,74,147,61,34]

#chnsenticorp
Dee_5=[0,0,0,45,3,0,1,33,177,298,286]
Our_5=[91,217,64,199,237,119,24,27,23,15,7]
cnn_5=[108,232,92,190,206,73,39,32,23,19,9]


'''
total_width, n = 0.8, 4
width = total_width / n
x = x - width*2
#plt.figure(figsize=(6,6))

plt.subplot(131)
#plt.tight_layout()
plt.bar(x, Dee_1,  width=width, label='DeeBERT',color='#427DA6')
plt.bar(x + width, RightTool_1,width=width, label='RightTool', color='orange')
plt.bar(x + 2 * width,Rome_1, width=width, label='RomeBERT', color='g')
plt.bar(x + 3 * width, Ode_1, width=width, label='OdeBERT',color='r')
plt.grid(ls='--',axis='y')
title='SNIPS'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(132)
#plt.tight_layout()
plt.bar(x, Dee_2, width=width, label='DeeBERT', color='#427DA6')
plt.bar(x + width, RightTool_2,width=width, label='RightTool', color='orange')
plt.bar(x + 2 * width,Rome_2,width=width, label='RomeBERT', color='g')
plt.bar(x + 3 * width, Ode_2,width=width, label='OdeBERT', color='r')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')

title='ECDT'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)
plt.grid(ls='--',axis='y')
plt.subplot(133)
#plt.tight_layout()
plt.bar(x, Dee_3, width=width, label='DeeBERT', color='#427DA6')
plt.bar(x + width, RightTool_3,width=width, label='RightTool', color='orange')
plt.bar(x + 2 * width,Rome_3, width=width, label='RomeBERT',color='g')
plt.bar(x + 3 * width, Ode_3,width=width, label='OdeBERT', color='r')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')
plt.grid(ls='--',axis='y')
title='FDQuestion'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)
'''
plt.subplot(154)
#plt.tight_layout()
plt.bar(x, Dee_4,  width=width, label='DeeBERT')
plt.bar(x + width, Our_4, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')

title='Shopping(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(155)
#plt.tight_layout()
plt.bar(x, Dee_5,  width=width, label='DeeBERT')
plt.bar(x + width, Our_5, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')

title='Chnsenticorp(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(223)
#plt.tight_layout()
plt.bar(x, Dee_3,  width=width, label='DeeBERT')
plt.bar(x + width, Our_3, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_3, width=width, label='Our+cnn')

title='Shopping(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(224)
#plt.tight_layout()
plt.bar(x, Dee_4,  width=width, label='DeeBERT')
plt.bar(x + width, Our_4, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_4, width=width, label='Our+cnn')

title='Chnsenticorp(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)
'''
plt.savefig("sample_layer.svg")
plt.show()