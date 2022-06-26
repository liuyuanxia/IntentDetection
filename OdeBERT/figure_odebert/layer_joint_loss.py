import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.rc('font',family='Times New Roman')
x = [1,2,3,4,5,6,7,8,9,10,11,12]
plt.figure(figsize=(16,5))
#DeeBERT
ASC_1=[61.77,51.57,26.54,52.47,34.03,17.24,20.69,20.84,24.74,41.38,84.86,95.65]
DEC_1=[61.77,52.47,27.59,53.07,34.18,17.24,20.68,24.44,24.44,40.63,82.46,95.65]
Norm_1=[61.77,52.47,27.59,53.07,34.33,17.54,20.69,20.84,24.74,41.38,84.86,95.65]

#RightTool
ASC_2=[30.58,34.48,45.58,77.21,94.75,95.35,95.50,95.50,95.35,95.35,94.90,95.05]
DEC_2=[87.86,92.50,93.85,94.00,94.60,94.30,94.15,94.30,94.30,94.00,94.00,94.00]
Norm_2=[74.96,88.91,92.80,93.70,94.60,94.30,94.15,94.15,94.00,94.15,94.45,94.45]

#RomeBERT
ASC_3=[75.26,90.85,93.7,94.75,95.2,95.35,95.05,95.5,95.2,95.2,95.2,95.2]
DEC_3=[85.61,88.46,89.81,90.55,90.70,90.40,90.55,90.70,91,90.85,90.7,90.4]
Norm_3=[83.96,90.4,92.35,92.95,93.55,93.7,94.15,94.15,94.15,94,94,94.3]

#OdeBERT
ASC_4=[86.21,92.80,93.55,94.60,95.05,95.20,95.20,95.65,95.50,95.50,95.65,95.20]
DEC_4=[91.45,94.3,94.3,94.15,94.6,95.35,94.9,95.2,95.5,95.5,95.2,94.9]
Norm_4=[95.95,95.8,96.4,96.25,95.8,95.8,95.95,94.6,93.7,93.7,92.95,88.91][::-1]

'''
ax =plt.subplot(221)
#plt.tight_layout()
plt.plot(x, ASC_1,marker='>', ms=6,label='ASC')
plt.plot(x, DEC_1, marker='h', ms=6,label='DEC')
plt.plot(x, Norm_1, marker='v', ms=6,label='Norm')
title='DeeBERT'
plt.title(title)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=7)
plt.grid(ls='--')
'''
ax =plt.subplot(131)
#plt.tight_layout()
plt.plot(x, ASC_2,marker='>', ms=6,label='ASC')
plt.plot(x, DEC_2, marker='h', ms=6,label='DEC')
plt.plot(x, Norm_2, marker='v', ms=6,label='Norm')
title='RightTool'
plt.title(title)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
plt.legend(fontsize=7)
plt.grid(ls='--')

ax=plt.subplot(132)
#plt.tight_layout()
plt.plot(x, ASC_3,marker='>', ms=6,label='ASC')
plt.plot(x, DEC_3, marker='h', ms=6,label='DEC')
plt.plot(x, Norm_3, marker='v', ms=6,label='Norm')
title='RomeBERT'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
plt.legend(fontsize=7)
plt.grid(ls='--')

ax=plt.subplot(133)
#plt.tight_layout()
plt.plot(x, ASC_4,marker='>', ms=6,label='ASC')
plt.plot(x, DEC_4, marker='h', ms=6,label='DEC')
plt.plot(x, Norm_4, marker='v', ms=6,label='Norm')
title='OdeBERT'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Layers')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.ylabel('Acc')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
plt.legend(fontsize=7)
plt.grid(ls='--')

#plt.savefig("./ACC+FLOPs.png")
plt.savefig("weight_layer_3.svg")
plt.show()
