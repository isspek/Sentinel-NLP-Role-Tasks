from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from utils import logger, report_results
from joblib import dump, load


def svm(X_train, y_train, X_dev, y_dev, X_test, y_test, random_state, feat_name):
    model_path = 'models/svm_{random_state}_{feat_name}.joblib'.format(random_state=random_state,
                                                                       feat_name=feat_name)
    if Path(model_path).exists():
        logger.info('{} exists already!!'.format(model_path))
        clf = load(model_path)
        y_pred = clf.predict(X_dev)
        y_pred_test = clf.predict(X_test)
        report_results(y_dev, y_pred, y_pred_test, y_test)
        return
    clf = LinearSVC(random_state=random_state, tol=1e-5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    y_pred_test = clf.predict(X_test)
    report_results(y_dev, y_pred, y_pred_test, y_test)

    dump(clf, model_path)
    logger.info('Saved to {}'.format(model_path))


def random_forest(X_train, y_train, X_dev, y_dev, X_test, y_test, random_state, feat_name):
    model_path = 'models/random_forest_{random_state}_{feat_name}.joblib'.format(random_state=random_state,
                                                                              feat_name=feat_name)
    if Path(model_path).exists():
        logger.info('{} exists already!!'.format(model_path))
        clf = load(model_path)
        y_pred = clf.predict(X_dev)
        y_pred_test = clf.predict(X_test)
        report_results(y_dev, y_pred, y_pred_test, y_test)
        return
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    y_pred_test = clf.predict(X_test)
    report_results(y_dev, y_pred, y_pred_test, y_test)

    dump(clf, model_path)
    logger.info('Saved to {}'.format(model_path))

MODELS = {
    'svm': svm,
    'random_forest': random_forest
}
