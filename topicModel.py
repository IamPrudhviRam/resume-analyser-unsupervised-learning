
import pandas as pd
import FeatureEngineeringStrategies as featureEngineeringStrategies
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(featureEngineeringStrategies.norm_corpus)
cv_matrix = cv_matrix.toarray()
print("cv_matrix",cv_matrix)

# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
pd.DataFrame(cv_matrix, columns=vocab)
print("pd",pd.DataFrame(cv_matrix, columns=vocab))

# you can set the n-gram range to 1,2 to get unigrams as well as bigrams
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(featureEngineeringStrategies.norm_corpus)

bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
pd.DataFrame(bv_matrix, columns=vocab)
print("pd",pd.DataFrame(bv_matrix, columns=vocab))

#-----------------------------------------------
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
print("features\n",features)

tt_matrix = lda.components_
for topic_weights in tt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 0.6]
    print("topic",topic)
    print()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=0)
km.fit_transform(features)
cluster_labels = km.labels_
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([featureEngineeringStrategies.corpus_df, cluster_labels], axis=1)
print("Kmeans\n",pd.concat([featureEngineeringStrategies.corpus_df, cluster_labels], axis=1))