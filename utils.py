import logging
import tldextract
import string
import json
import pickle
import random
import re
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from diskcache import Cache
import argparse
from pathlib import Path

CACHE = Cache(Path('tmp'))

# logger settings
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = 0

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('main.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

DATA_FOLDER = Path('data')


def extract_host(url: str):
    ext = tldextract.extract(url)
    host = ext.domain
    return host


def get_task1_data(mode: str):
    assert mode == 'train' or mode == 'test'
    with open(DATA_FOLDER / 'fake_{mode}.json'.format(mode=mode)) as f:
        fake = json.load(f)

    with open(DATA_FOLDER / 'real_{mode}.json'.format(mode=mode)) as f:
        real = json.load(f)

    return {'fake': fake, 'real': real}


def split_by_sources(train_df):
    splits_fname = 'data_splits_task1.p'
    splits_fpath = DATA_FOLDER / splits_fname

    if splits_fpath.exists():
        with open(splits_fpath, 'rb') as fp:
            data = pickle.load(fp)
            return data

    train_df['host'] = train_df.url.map(extract_host)
    hosts = list(train_df['host'].unique())
    num_hosts_train = round(len(hosts) * 0.8)
    secure_random = random.SystemRandom()
    secure_random.seed(42)
    train_hosts = []

    while len(train_hosts) < num_hosts_train:
        choice = secure_random.choice(hosts)
        if choice not in train_hosts:
            train_hosts.append(choice)

    dev_hosts = list(set(hosts) - set(train_hosts))

    dev_df = train_df[train_df['host'].isin(dev_hosts)]
    train_df = train_df[train_df['host'].isin(train_hosts)]

    data = {'train_title': train_df['title'],
            'train_text': train_df['text'],
            'dev_title': dev_df['title'],
            'dev_text': dev_df['text']
            }

    with open(splits_fpath, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def get_task2_data():
    with open(DATA_FOLDER / 'clusters.json') as f:
        cluster = json.load(f)
    return cluster


def report_results(y_dev, y_pred, y_pred_test, y_test):
    logger.info('Classification Report - Development Set')
    logger.info(classification_report(y_true=y_dev, y_pred=y_pred))
    logger.info('ROC-AUC Score')
    logger.info(roc_auc_score(y_true=y_dev, y_score=y_pred))
    logger.info('Classification Report - Test Set')
    logger.info(classification_report(y_true=y_test, y_pred=y_pred_test))
    logger.info('ROC-AUC Score')
    logger.info(roc_auc_score(y_true=y_test, y_score=y_pred_test))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub(r"[\n\t\s]+", " ", text)
    text = text.strip()
    return text