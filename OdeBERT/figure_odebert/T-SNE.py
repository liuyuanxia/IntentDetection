import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

layer1="features_our_1.txt"
our_features1=np.loadtxt(layer1)
layer3="features_our_3.txt"
our_features3=np.loadtxt(layer3)
layer6="features_our_6.txt"
our_features6=np.loadtxt(layer6)
layer9="features_our_9.txt"
our_features9=np.loadtxt(layer9)
layer12="features_our_12.txt"
our_features12=np.loadtxt(layer12)


from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(figsize=(10,3))
fig.suptitle('ECDT')
#plt.figure(figsize=)

#plt.axis('off')

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(our_features1.data)
font = {"color": "k",
        "size": 13,
        "family" : "serif"}

plt.subplot(1, 5, 1)
plt.tight_layout()
plt.title('Layer1')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  alpha=0.6 ,
            cmap=plt.cm.get_cmap('rainbow', 10))

#
plt.ylabel('our')


X_tsne = TSNE(n_components=2, random_state=33).fit_transform(our_features3.data)
font = {"color": "k",
        "size": 13,
        "family" : "serif"}

plt.subplot(1, 5, 2)
plt.tight_layout()
plt.title('Layer3')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))


X_tsne = TSNE(n_components=2, random_state=33).fit_transform(our_features6.data)
font = {"color": "k",
        "size": 13,
        "family" : "serif"}
plt.subplot(1, 5, 3)
plt.tight_layout()
plt.title('Layer6')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(our_features9.data)
font = {"color": "k",
        "size": 13,
        "family" : "serif"}

plt.subplot(1, 5, 4)
plt.tight_layout()
plt.title('Layer9')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))


X_tsne = TSNE(n_components=2, random_state=33).fit_transform(our_features12.data)
font = {"color": "k",
        "size": 13,
        "family" : "serif"}

plt.subplot(1, 5, 5)
plt.tight_layout()
plt.title('Layer12')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  alpha=0.6,
            cmap=plt.cm.get_cmap('rainbow', 10))

plt.show()