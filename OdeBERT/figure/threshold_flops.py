import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
#plt.figure(dpi=200,figsize=(16,8))
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
DeeBERT=[1.117,1.173,1.231,1.326,1.467,1.743,2.423,4.607,11.04]
OeBERT=[3.446,4.412,5.329,6.205,7.174,8.289,9.366,10.98,11.86]
plt.subplot(132)
#plt.tight_layout()
plt.plot(x, DeeBERT,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, OeBERT, marker='h', ms=5,label='OeBERT')
#plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='ECDT'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Threshold')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.ylabel('Spe')
plt.legend(fontsize=12)


x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
DeeBERT=[8.262,9.317,10.46,11.35,11.57,11.82,11.94,11.99,11.99]
OeBERT=[8.575,9.66,10.88,11.42,11.75,11.82,11.94,11.99,11.99]
plt.subplot(131)
#plt.tight_layout()
plt.plot(x, DeeBERT,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, OeBERT, marker='h', ms=5,label='OeBERT')
#plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='SNIPS'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Threshold')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.ylabel('Spe')
plt.legend(fontsize=12)

x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
DeeBERT=[1.008,1.024,1.049,1.084,1.135,1.259,1.656,3.606,11.31]
OeBERT=[1.713,2.269,3.009,3.926,4.984,6.310,7.84,9.576,11.481]
plt.subplot(133)
#plt.tight_layout()
plt.plot(x, DeeBERT,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, OeBERT, marker='h', ms=5,label='OeBERT')
#plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='FDQuestion'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Threshold')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.ylabel('Spe')
plt.legend(fontsize=12)
'''

x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
DeeBERT=[1.725,2.078,2.424,2.821,2.936,3.933,4.744,5.886,7.687]
OeBERT=[4.419,5.507,6.397,7.179,7.857,8.562,9.245,9.938,10.74]
plt.subplot(223)
#plt.tight_layout()
plt.plot(x, DeeBERT,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, OeBERT, marker='h', ms=5,label='OeBERT')
#plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='Shopping'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Threshold')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.ylabel('Speedup')
plt.legend(fontsize=8)

x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
DeeBERT=[1.154,1.274,1.391,1.535,1.733,2.03,2.404,3.142,4.834]
OeBERT=[2.264,2.936,3.619,4.270,4.971,5.826,6.827,7.885,9.266]
plt.subplot(224)
#plt.tight_layout()
plt.plot(x, DeeBERT,marker='x', ms=5,mec='r', mfc='w',label='DeeBERT')
plt.plot(x, OeBERT, marker='h', ms=5,label='OeBERT')
#plt.plot(x, LMCL_1[::-1], marker='v', ms=5,label='OdeBERT')
title='Chnsenticorp'
plt.title(title)
#plt.ylim(-5,350)
#plt.title('line chart')
plt.xlabel('Threshold')
plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.ylabel('Speedup')
plt.legend(fontsize=8)
'''
plt.savefig("./Threshold+FLOPs.png")
plt.show()