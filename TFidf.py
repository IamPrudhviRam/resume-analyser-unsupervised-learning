from sklearn.feature_extraction.text import TfidfVectorizer
import main as main
import pandas as pd

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(main.documents)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.to_csv('resume_dataset.csv')