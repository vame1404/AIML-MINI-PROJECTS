import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix








plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()