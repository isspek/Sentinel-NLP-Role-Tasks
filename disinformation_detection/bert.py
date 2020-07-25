import torch
from utils import logger, report_results
import numpy as np
import random
from transformers import BertTokenizer
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import TensorDataset
from pathlib import Path


def bert_encoder(articles, tokenizer):
    ids = []
    att_masks = []
    for article in articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=250,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
        ids.append(encoded_article['input_ids'])
        att_masks.append(encoded_article['attention_mask'])

    ids = torch.cat(ids, dim=0)
    att_masks = torch.cat(att_masks, dim=0)
    return ids, att_masks


def get_torch_datasets(X_train, y_train, X_dev, y_dev, X_test, y_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_ids, train_att_masks = bert_encoder(X_train, tokenizer)
    dev_ids, dev_att_masks = bert_encoder(X_dev, tokenizer)
    test_ids, test_att_masks = bert_encoder(X_test, tokenizer)

    logger.info('Encoding of samples are completed.')

    y_train = torch.tensor(y_train)
    y_dev = torch.tensor(y_dev)
    y_test = torch.tensor(y_test)

    logger.info('Encoding of labels are completed.')

    train_dataset = TensorDataset(train_ids, train_att_masks, y_train)
    test_dataset = TensorDataset(test_ids, test_att_masks, y_test)
    dev_dataset = TensorDataset(dev_ids, dev_att_masks, y_dev)

    logger.info('Torch datasets are ready.')
    return train_dataset, dev_dataset, test_dataset


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def bert_train_helper(train_dataset: object, dev_dataset: object, test_dataset: object,
                      device: object, random_seed: int,
                      model_params: dict):
    lr = model_params['lr']
    epochs = model_params['epochs']
    batch_size = model_params['batch_size']

    model_name = ''
    for key, value in model_params.items():
        model_name += key + '_' + str(value) + '_'
    model_name = model_name + 'device_{}'.format(device)

    checkpoint_dir = Path('models') / model_name

    set_seed(random_seed)
    torch.cuda.empty_cache()
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=False,
                                                          output_hidden_states=False)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    train_iter = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    dev_iter = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)
    test_iter = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    logger.info(
        "Number of training samples {train}, number of test samples {test}, number of dev {dev} samples".format(
            train=len(train_dataset),
            test=len(test_dataset),
            dev=len(dev_dataset)))

    if not checkpoint_dir.exists():
        device = torch.device(device)
        total_iter = len(train_iter) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

        ################### Training ###########################
        for epoch in range(epochs):
            model.to(device)
            model.train()

            logger.info('Training epoch: {}'.format(epoch + 1))
            train_loss = 0

            preds = []
            trues = []

            for batch_ids in tqdm(train_iter):
                input_ids = batch_ids[0].to(device)
                att_masks = batch_ids[1].to(device)
                labels = batch_ids[2].to(device)

                model.zero_grad()

                # forward pass
                loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

                _pred = logits.cpu().data.numpy()
                preds.append(_pred)
                _label = labels.cpu().data.numpy()
                trues.append(_label)
                train_loss += loss.item()

                # backpropagate and update optimizer learning rate
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            save_model(model, checkpoint_dir)

    ################### Evaluation ###########################
    model = load_model(checkpoint_dir=checkpoint_dir)
    y_test, y_pred_test = inference(test_iter, model, device)
    y_dev, y_pred = inference(dev_iter, model, device)
    logger.info(y_pred_test)
    report_results(y_dev=y_dev, y_pred=y_pred, y_pred_test=y_pred_test, y_test=y_test)
    del model


def save_model(model, checkpoint_dir):
    model.save_pretrained(checkpoint_dir)
    return


def load_model(checkpoint_dir):
    model = BertForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=2,
                                                          output_attentions=False, output_hidden_states=False)
    return model


def inference(test_iter, model, device):
    device = torch.device(device)
    model.to(device)
    model.eval()
    preds = []
    trues = []
    for batch_ids in tqdm(test_iter):
        input_ids = batch_ids[0].to(device)
        att_masks = batch_ids[1].to(device)
        labels = batch_ids[2].to(device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    trues = np.concatenate(trues)
    preds = np.concatenate(preds)
    preds = np.array([np.argmax(numarray, axis=0) for numarray in preds])
    return trues, preds
