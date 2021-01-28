import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import main as main

pd.options.display.max_colwidth = 200

corpus=main.documents
# print("corpus",corpus)
#---------------------------------------------
labels = ['java','angular','reactjs','python','ui/ux','tester','business analyst',
          'java','angular','reactjs','python']

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus,
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]

print("corpus_df",corpus_df)
#-----------------------------------------------

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(corpus)
print("norm_corpus",norm_corpus)

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0.5, max_df=1.0, use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab).to_csv('tfidf.csv')
print("pd",pd.DataFrame(np.round(tv_matrix, 5), columns=vocab))
#-----------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
print("similarity_df",similarity_df)
similarity_df.to_csv("similarity_df.csv")
#-----------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_matrix, 'ward')
pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2',
                         'Distance', 'Cluster Size'], dtype='object')

print(pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2',
                         'Distance', 'Cluster Size'], dtype='object'))

#-------------------------------------------------

plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
plt.show()
#--------------------------------------------------
from scipy.cluster.hierarchy import fcluster
max_dist = 1.0

cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
pd.concat([corpus_df, cluster_labels], axis=1)
print(pd.concat([corpus_df, cluster_labels], axis=1))