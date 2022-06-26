import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


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
Dee_1=[561,89,22,13,1,1,0,1,0,1,2]
Our_1=[620,46,9,3,3,0,0,3,1,0,0]
cnn_1=[621,33,13,10,4,0,1,2,1,0,0]

#UserIntent
Dee_2=[57,1,0,4,1,0,0,0,0,42,84]
Our_2=[121,167,124,108,59,21,17,10,8,5,4]
cnn_2=[334,118,53,64,33,12,9,5,6,7,1]

#Shopping
Dee_3=[189,132,138,3160,1095,52,128,1421,1563,523,520]
Our_3=[4742,2321,715,637,437,241,123,88,123,56,5]
cnn_3=[5092,1252,753,702,423,145,105,74,147,61,34]

#chnsenticorp
Dee_4=[0,0,0,45,3,0,1,33,177,298,286]
Our_4=[91,217,64,199,237,119,24,27,23,15,7]
cnn_4=[108,232,92,190,206,73,39,32,23,19,9]
#FDQ
Dee_5=[0,0,0,0,0,0,0,0,0,4,113]
Our_5=[5,63,112,201,164,83,51,26,19,12,5]


total_width, n = 0.9, 3
width = total_width / n
x = x - (total_width - width) / 2
#plt.figure(figsize=(6,6))

plt.subplot(131)
#plt.tight_layout()
plt.bar(x, Dee_1,  width=width, label='DeeBERT')
plt.bar(x + width, Our_1, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_1, width=width, label='Our+cnn')

title='SNIPS(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(132)
#plt.tight_layout()
plt.bar(x, Dee_2,  width=width, label='DeeBERT')
plt.bar(x + width, Our_2, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')

title='ECDT(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)

plt.subplot(133)
#plt.tight_layout()
plt.bar(x, Dee_5,  width=width, label='DeeBERT')
plt.bar(x + width, Our_5, width=width, label='OeBERT')
#plt.bar(x + 2 * width, cnn_2, width=width, label='Our+cnn')

title='FDQuestion(T=0.1)'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Exiting samples')
plt.legend(fontsize=7)
'''
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
plt.show()