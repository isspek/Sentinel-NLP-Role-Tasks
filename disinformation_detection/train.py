from utils import get_task1_data, logger, split_by_sources
from .feature_extractor import FEATURES, ngram
from .bert import get_torch_datasets, bert_train_helper
import torch
import pandas as pd
import numpy as np
import argparse
from .linear_models import MODELS
from utils import str2bool

SEP = '<-SEP->'


def bert_train(random_seed: int):
    full_dev_texts, full_dev_titles, full_test_texts, full_test_titles, full_train_texts, full_train_titles, y_dev, y_test, y_train = read_data()

    X_train = []
    for i in range(len(full_train_titles)):
        X_train.append(full_train_titles[i] + SEP + full_train_texts[i])
    X_train = np.asarray(X_train)

    X_dev = []
    for i in range(len(full_dev_titles)):
        X_dev.append(full_dev_titles[i] + SEP + full_dev_texts[i])
    X_dev = np.asarray(X_dev)

    X_test = []
    for i in range(len(full_test_titles)):
        X_test.append(full_test_titles[i] + SEP + full_test_texts[i])
    X_test = np.asarray(X_test)

    train_dataset, dev_dataset, test_dataset = get_torch_datasets(X_train, y_train, X_dev, y_dev, X_test,
                                                                  y_test)
    device = 'cuda' if args.use_gpu and torch.cuda.is_available else 'cpu'
    model_params = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    bert_train_helper(train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
                      random_seed=random_seed, device=device, model_params=model_params)


def feats_train(random_seed: int, model: str, features: list):
    '''

    :param random_seed: seed value for reproducibility
    :param model: the name of model choice e.g svm, naive, BERT
    :param features: features that are used as an input
    :return:
    '''
    full_dev_texts, full_dev_titles, full_test_texts, full_test_titles, full_train_texts, full_train_titles, y_dev, y_test, y_train = read_data()

    if len(features) == 1:
        _feature = FEATURES[features[0]]
        if _feature.__name__ == 'ngram':
            X_train, X_dev, X_test = ngram(full_train_texts, full_dev_texts, full_test_texts)
        else:
            X_train = np.asarray([_feature(text) for text in full_train_texts])
            X_dev = np.asarray([_feature(text) for text in full_dev_texts])
            X_test = np.asarray([_feature(text) for text in full_test_texts])

    else:
        X_train = np.array([], dtype=np.float).reshape(len(full_train_titles), 0)
        X_dev = np.array([], dtype=np.float).reshape(len(full_dev_titles), 0)
        X_test = np.array([], dtype=np.float).reshape(len(full_test_titles), 0)

        for feature in features:
            if feature == 'ngram':
                _X_train, _X_dev, _X_test = ngram(full_train_texts, full_dev_texts, full_test_texts)
                X_train = np.concatenate((X_train, _X_train),axis=1)
                X_dev = np.concatenate((X_dev, _X_dev), axis=1)
                X_test = np.concatenate((X_test, _X_test), axis=1)
            else:
                X_train = np.concatenate((X_train, np.asarray([FEATURES[feature](text) for text in full_train_texts])),
                                         axis=1)
                X_dev = np.concatenate((X_dev, np.asarray([FEATURES[feature](text) for text in full_dev_texts])),
                                       axis=1)
                X_test = np.concatenate((X_test, np.asarray([FEATURES[feature](text) for text in full_test_texts])),
                                        axis=1)

    assert len(X_train) == len(y_train)
    assert len(X_dev) == len(y_dev)

    feat_name = '_'.join(i for i in features)
    MODELS[model](X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev, X_test=X_test, y_test=y_test,
                  random_state=random_seed,
                  feat_name=feat_name)

def read_data():
    train = get_task1_data(mode='train')
    fake_train = pd.DataFrame(train['fake'])
    real_train = pd.DataFrame(train['real'])
    fake_train_dict = split_by_sources(fake_train)
    fake_train_titles = fake_train_dict['train_title'].tolist()
    fake_train_texts = fake_train_dict['train_text'].tolist()
    fake_dev_titles = fake_train_dict['dev_title'].tolist()
    fake_dev_texts = fake_train_dict['dev_text'].tolist()
    real_train_dict = split_by_sources(real_train)
    real_train_titles = real_train_dict['train_title'].tolist()
    real_train_texts = real_train_dict['train_text'].tolist()
    real_dev_titles = real_train_dict['dev_title'].tolist()
    real_dev_texts = real_train_dict['dev_text'].tolist()
    logger.info('==========Stats==========')
    logger.info('Training')
    logger.info('Fake samples {}'.format(len(fake_train_titles)))
    logger.info('Real samples {}'.format(len(real_train_titles)))
    logger.info('Development')
    logger.info('Fake samples {}'.format(len(fake_dev_titles)))
    logger.info('Real samples {}'.format(len(real_dev_titles)))
    full_train_titles = fake_train_titles + real_train_titles
    full_dev_titles = fake_dev_titles + real_dev_titles
    full_train_texts = fake_train_texts + real_train_texts
    full_dev_texts = fake_dev_texts + real_dev_texts
    y_train = [1 for _ in range(0, len(fake_train_titles))] + [0 for _ in range(0, len(real_train_titles))]
    y_dev = [1 for _ in range(0, len(fake_dev_titles))] + [0 for _ in range(0, len(real_dev_titles))]
    test = get_task1_data(mode='test')
    fake_test = pd.DataFrame(test['fake'])
    real_test = pd.DataFrame(test['real'])
    fake_test_titles = fake_test['title'].tolist()
    fake_test_texts = fake_test['text'].tolist()
    real_test_titles = real_test['title'].tolist()
    real_test_texts = real_test['text'].tolist()
    logger.info('==========Stats==========')
    logger.info('Test')
    logger.info('Fake samples {}'.format(len(fake_test_titles)))
    logger.info('Real samples {}'.format(len(real_test_titles)))
    full_test_titles = fake_test_texts + real_test_titles
    full_test_texts = fake_test_texts + real_test_texts
    y_test = [1 for _ in range(0, len(fake_test_titles))] + [0 for _ in range(0, len(real_test_titles))]
    return full_dev_texts, full_dev_titles, full_test_texts, full_test_titles, full_train_texts, full_train_titles, y_dev, y_test, y_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ######### Feature engineering model params ##############################
    parser.add_argument('--random_seed', type=int, required=True)
    parser.add_argument('--model', type=str, choices=['svm', 'random_forest', 'bert'], required=True)
    parser.add_argument("--features", nargs="+")
    ######### BERT model params ##############################
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--use_gpu", type=str2bool)

    args = parser.parse_args()

    if args.model == 'bert':
        bert_train(random_seed=args.random_seed)
    else:
        feats_train(random_seed=args.random_seed, model=args.model, features=args.features)
