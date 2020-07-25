from utils import get_task1_data, logger, extract_host
import pandas as pd

train = get_task1_data(mode='train')

fake_train = train['fake']
real_train = train['real']

logger.info('Number of fake samples in train set {}'.format(len(fake_train)))
logger.info('Number of real samples in train set {}'.format(len(real_train)))
logger.info('Total samples in train set {}'.format(len(fake_train + real_train)))

fake_train_df = pd.DataFrame(fake_train)
fake_train_df['host'] = fake_train_df.url.map(extract_host)
logger.info(
    'Number of unique hosts dissemination disinformation in train set {}'.format(len(fake_train_df.host.unique())))
logger.info(fake_train_df.groupby(['host'])['url'].count().nlargest(10))
real_train_df = pd.DataFrame(real_train)
real_train_df['host'] = real_train_df.url.map(extract_host)
logger.info(
    'Number of unique hosts dissemination disinformation in train set {}'.format(len(real_train_df.host.unique())))
logger.info(real_train_df.groupby(['host'])['url'].count().nlargest(10))

