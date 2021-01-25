import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import TFidf as tf
sil = []
kl = []
kmax = 10
for k in range(3, kmax+1):
    kmeans2 = KMeans(n_clusters=k,random_state=700).fit(tf.denselist)
    labels = kmeans2.labels_
    sil.append(silhouette_score(tf.denselist, labels, metric='euclidean'))
    kl.append(k)
print("sil",sil)
print("kl",kl)

plt.plot(kl, sil)
plt.title("Resume_Analyser")
plt.ylabel('Silhoutte Score')
plt.xlabel('K')
plt.show()