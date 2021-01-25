from sklearn.feature_extraction.text import TfidfVectorizer
import main as main


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(main.documents)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
