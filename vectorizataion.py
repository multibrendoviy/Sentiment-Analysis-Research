import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD



MAX_WORDS = 5000
RAND = 10

class DenseCountfVectorizer(CountVectorizer):
    """
    Класс наследующий от CountfVectorizer, преобразующий разреженную матрицу
    в плотную для использования с логистической регрессией.
    """
    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        X = pd.DataFrame(X.toarray())
        return X

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y=y)
        X = pd.DataFrame(X.toarray())
        return X


class DenseTfidfVectorizer(TfidfVectorizer):
    """
    Класс наследующий от TfidfVectorizer, преобразующий разреженную матрицу
    в плотную для использования с логистической регрессией.
    """
    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        X = pd.DataFrame(X.toarray())
        return X

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y=y)
        X = pd.DataFrame(X.toarray())
        return X

    
class TfIdfSVDTransformer(TransformerMixin):
    """
    Класс TfIdfSVDTransformer настраивает и применяет преобразование TF-IDF 
    и сокращение размерности с помощью SVD с предварительной старндартизацией.

    Атрибуты:
    - tfidf_max_features: Максимальное количество признаков для TF-IDF.
    - svd_n_components: Количество компонентов для сокращения размерности.
    - svd_n_iter: Количество итераций для SVD.

    Методы:
    - fit(X, y=None): Обучает TF-IDF, масштабирует данные и применяет сокращение размерности.
    - transform(X, y=None): Применяет обученное преобразование к данным.
    """
    def __init__(self, tfidf_max_features=MAX_WORDS, svd_n_components=200, 
                 svd_n_iter=100, random_state=RAND):
        self.tfidf_max_features = tfidf_max_features
        self.svd_n_components = svd_n_components
        self.svd_n_iter = svd_n_iter
        self.random_state = random_state
        # Преобразования данных в формат TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=
                                                self.tfidf_max_features)
        # Стандартизация перед снижение размерности
        self.scaler = StandardScaler()
        # Снижение размерности данных методом сингулярного разложения матрицы документ-слово
        self.lsa = TruncatedSVD(n_components=self.svd_n_components, 
                                n_iter=self.svd_n_iter, 
                                random_state=self.random_state)

    def fit(self, X:np.ndarray, y=None):
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        X_tfidf = pd.DataFrame(X_tfidf.toarray())
        X_scaled = self.scaler.fit_transform(X_tfidf)
        self.lsa.fit(X_scaled)
        return self
    



    def transform(self, X:np.ndarray, y=None):
        X_tfidf = self.tfidf_vectorizer.transform(X)
        X_tfidf = pd.DataFrame(X_tfidf.toarray())
        X_scaled = self.scaler.transform(X_tfidf)
        return self.lsa.transform(X_scaled)
    
    
class mean_vectorizer(TransformerMixin):
    """
    Класс для векторизации данных в формате Word2Vec, документ будет представлять 
    из себя усредненный вектор векторов слов, входящих в документ
    """
    def __init__(self, word2vec, dim=300):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        return np.array([np.mean([self.word2vec[w] for w in doc.split()   \
                   if w in self.word2vec] or \
                       [np.zeros(self.dim)], axis=0) for doc in X])