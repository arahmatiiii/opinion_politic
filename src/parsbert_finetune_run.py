'''
parsbert_finetune_run.py is written for for finetune bert
'''


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas
import torch
from transformers import DistilBertForSequenceClassification, Trainer, \
      TrainingArguments, BertForSequenceClassification, BertForSequenceClassification
from transformers import DistilBertTokenizerFast, BertTokenizer


def load_dataset():
    """
    method to load data
    :return: datasets texr and labels by list
    """
    train_data = pandas.read_csv('../data/Processed/train_data_normed.csv')
    train_texts = list(train_data.text)
    train_labels = list(train_data.label)

    test_data = pandas.read_csv('../data/Processed/test_data_normed.csv')
    test_texts = list(test_data.text)
    test_labels = list(test_data.label)

    valid_data = pandas.read_csv('../data/Processed/valid_data_normed.csv')
    valid_texts = list(valid_data.text)
    valid_labels = list(valid_data.label)

    return train_texts, train_labels, test_texts, test_labels, valid_texts, valid_labels


class Dataset(torch.utils.data.Dataset):
    """
    class for create dataset for train bert
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    method to compute metrics
    :param pred:
    :return: dic of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    train_texts, train_labels, test_texts, \
    test_labels, valid_texts, valid_labels = load_dataset()

    model_name = '../models/parsbert_models/'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_labels)
    test_dataset = Dataset(test_encodings, test_labels)


    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1500,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics

    )

    trainer.train()
    print(trainer.evaluate())