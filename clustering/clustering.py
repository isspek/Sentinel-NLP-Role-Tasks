import pandas as pd
from utils import get_task2_data, logger, str2bool, preprocess
from pathlib import Path
import argparse
import string
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import homogeneity_score

MODEL_FOLDER = Path('models')
LEMMATIZER = WordNetLemmatizer()


def preprocess_clustering(text: str):
    text = preprocess(text)
    tokens = text.split(' ')
    doc = []
    for token in tokens:
        if token in string.punctuation:
            continue
        if token.isnumeric():
            continue
        if len(token) < 2:
            continue

        # lemmatize the words
        token = LEMMATIZER.lemmatize(token)
        doc.append(token)
    return doc


def doc2vec_helper(doc, model, size=300):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in doc:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def doc2vec(docs, model_name):
    doc2vec_path = MODEL_FOLDER / model_name

    if not doc2vec_path.exists():
        docs = [preprocess_clustering(content) for content in docs]
        pretrained_embeddings_path = Path('data') / 'GoogleNews-vectors-negative300.bin'

        model = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
        X_doc2vecs = np.asarray([doc2vec_helper(doc, model) for doc in docs])
        X_doc2vecs = X_doc2vecs.reshape((X_doc2vecs.shape[0], 300))

        with open(doc2vec_path, 'wb') as file:
            pickle.dump(X_doc2vecs, file)

    with open(doc2vec_path, 'rb') as file:
        X_doc2vecs = pickle.load(file)
    return X_doc2vecs


def get_ner_vectors(text, nlp):
    doc = nlp(text)
    entity_tags = dict.fromkeys(['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART'],
                                0)
    for ent in doc.ents:
        if ent.label_ in entity_tags.keys():
            entity_tags[ent.label_] += 1
    entity_vec = np.array(list(entity_tags.values()))
    entity_vec = entity_vec.reshape(1, len(entity_tags.keys()))
    return entity_vec


def bert_vec(docs, model_name):
    bert_model_path = MODEL_FOLDER / model_name

    if not bert_model_path.exists():
        model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        embeddings = model.encode(docs)
        with open(bert_model_path, 'wb') as file:
            pickle.dump(embeddings, file)

    with open(bert_model_path, 'rb') as file:
        X_bert = pickle.load(file)
    return X_bert


def NERvec(docs, model_name):
    ner_model_path = MODEL_FOLDER / model_name

    if not ner_model_path.exists():
        nlp = spacy.load("en_core_web_sm")
        # docs = [preprocess_clustering(content) for content in data['text']]
        X_ner = np.array([get_ner_vectors(doc, nlp) for doc in docs])
        X_ner = X_ner.reshape((X_ner.shape[0], X_ner.shape[2]))
        min_max_scaler = MinMaxScaler()
        X_ner = min_max_scaler.fit_transform(X_ner)
        with open(ner_model_path, 'wb') as file:
            pickle.dump(X_ner, file)

    with open(ner_model_path, 'rb') as file:
        X_ner = pickle.load(file)

    return X_ner


def read_data():
    data = pd.DataFrame(get_task2_data())
    # columns ['id', 'text', 'title', 'lang', 'date', 'cluster', 'cluster_name'
    logger.info('Number of samples {}'.format(len(data)))
    logger.info('Language vs. articles {}'.format(data.groupby('lang')['id'].count()))
    expected_clusters_num = len(data['cluster_name'].unique())
    logger.info('Expected cluster {}'.format(expected_clusters_num))
    num_topics = expected_clusters_num
    return data, num_topics


def visualize(source, pos_x, pos_y, clusters):
    output_file('clustering.html')
    num_clusters = len(set(clusters))
    colors = np.array(sns.color_palette(sns.color_palette("muted", num_clusters)).as_hex())
    logger.info(len(colors))
    source = ColumnDataSource(dict(
        x=pos_x,
        y=pos_y,
        id=source['id'],
        title=source['title'],
        # text=source['text'],
        date=source['date'],
        cluster_name=source['cluster_name'],
        color=colors[clusters],
        label=['Cluster {}'.format(i) for i in clusters]
    ))
    plot = figure(title="Clustering with {} clusters".format(num_clusters),
                  plot_width=900, plot_height=700, tools="pan,wheel_zoom,box_zoom,reset,hover,save")

    plot.scatter(source=source, x='x', y='y', color='color', legend_group='label')

    # hover tools
    hover = plot.select(dict(type=HoverTool))

    hover.tooltips = {"content": "Id: @id, Title: @title, Date: @date, Cluster: @cluster_name"}
    plot.legend.location = "top_left"
    show(plot)


def clustering(params):
    data, num_topics = read_data()
    inputs = params.inputs
    if inputs == 'title':
        sub_data = data['title'].tolist()
    elif inputs == 'text':
        sub_data = data['text'].tolist()

    if params.model == 'doc2vec':
        model_name = 'doc2vec' + inputs + '.pk'
        X = doc2vec(sub_data, model_name)
    elif params.model == 'ner':
        model_name = 'ner' + inputs + '.pk'
        X = NERvec(sub_data, model_name)
    elif params.model == 'doc2vec_ner':
        model_name_ner = 'ner' + inputs + '.pk'
        model_name_doc2vec = 'doc2vec' + inputs + '.pk'
        X = np.concatenate((doc2vec(sub_data, model_name_doc2vec), NERvec(sub_data, model_name_ner)), axis=1)
        logger.info(X.shape)
    elif params.model == 'bert':
        assert 'title' == inputs
        model_name = 'berttitle.pk'
        X = bert_vec(sub_data, model_name)

    k = num_topics
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=1000, random_state=args.random_seed)
    y_pred = kmeans.fit_predict(X)

    logger.info('Silhoutte score: {}'.format(silhouette_score(X, y_pred, random_state=args.random_seed)))
    y_ground = [int(i) for i in data['cluster'].tolist()]
    logger.info('Homogeneity score: {}'.format(homogeneity_score(y_ground, y_pred)))

    pca = PCA(n_components=2, random_state=args.random_seed)
    reduced_pos = pca.fit_transform(X)
    visualize(data, reduced_pos[:, 0], reduced_pos[:, 1], y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, required=True)
    parser.add_argument('--model', type=str, choices=['doc2vec', 'ner', 'doc2vec_ner', 'bert'], required=True)
    parser.add_argument('--inputs', type=str, choices=['title', 'text'], required=True)

    args = parser.parse_args()

    clustering(args)
