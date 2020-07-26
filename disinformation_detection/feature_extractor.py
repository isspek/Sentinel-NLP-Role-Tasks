import nltk
import numpy as np
from nela_features.nela_features import NELAFeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import CACHE, preprocess

nltk.download('maxent_ne_chunker')
nltk.download('words')

nela = NELAFeatureExtractor()


@CACHE.memoize(tag='style')
def style_feat(text: str):
    style, _ = nela.extract_style(text)
    return style


@CACHE.memoize(tag='complexity')
def complexity_feat(text: str):
    complexity, _ = nela.extract_complexity(text)
    return complexity


@CACHE.memoize(tag='bias')
def bias_feat(text: str):
    bias, _ = nela.extract_bias(text)
    return bias


@CACHE.memoize(tag='affect')
def affect_feat(text: str):
    affect, _ = nela.extract_affect(text)
    return affect


@CACHE.memoize(tag='moral')
def moral_feat(text: str):
    moral, _ = nela.extract_moral(text)
    return moral


# @CACHE.memoize(tag='event')
# def event_feat(text: str):
#     event, _ = nela.extract_event(text)
#     return event

@CACHE.memoize(tag='cos_sim')
def cos_similarity(title: str, text: str):
    cos_sim = np.dot(title, text) / (np.norm(title) * np.norm(text))
    return cos_sim


def ngram(texts_train: list, texts_dev: list, texts_test: list):
    texts_train = np.array([preprocess(text) for text in texts_train])
    texts_dev = np.array([preprocess(text) for text in texts_dev])
    texts_test = np.array([preprocess(text) for text in texts_test])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(texts_train).toarray()
    X_dev = vectorizer.transform(texts_dev).toarray()
    X_test = vectorizer.transform(texts_test).toarray()
    return X_train, X_dev, X_test


FEATURES = {
    'style_feat': style_feat,
    'complexity_feat': complexity_feat,
    'bias_feat': bias_feat,
    'affect_feat': affect_feat,
    'moral_feat': moral_feat,
    # 'event_feat': event_feat,
    'cos_similarity': cos_similarity,
    'ngram': ngram
}
