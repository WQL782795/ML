import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, PreTrainedTokenizer, Trainer, TrainingArguments, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

class localDataSet(Dataset):
    def __init__(self, type, local_path):
        super().__init__()
        self.dataset = load_dataset(type, data_files=local_path)

    def __len__(self):
        return self.dataset.num_rows['train']

    def __getitem__(self, index):
        return self.dataset['train'][index]['text'], self.dataset['train'][index]['label']

def collate_fn(data):
    txt = [d[0] for d in data]
    label = [d[1] for d in data]
    return encode(txt)+ (torch.LongTensor(label),)


class SCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = BertModel.from_pretrained('bert-base-chinese')
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        self.classifier = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2),
                                        nn.Softmax())

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.base_model(input_ids, attention_mask, token_type_ids)
        return self.classifier(x.pooler_output)


def load_data(path, shuffle):
    dataset = localDataSet('arrow', path)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=shuffle, collate_fn=collate_fn)
    return train_loader


def encode(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    out = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text, padding='max_length', truncation=True,
                                      max_length=500, return_token_type_ids=True, return_tensors='pt',
                                      return_attention_mask=True, return_special_tokens_mask=True, return_length=True)
    return out['input_ids'], out['attention_mask'], out['token_type_ids']


def train(dataloader, epoch):
    sc = SCModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(sc.parameters(), lr=5e-4)
    sc.train()

    for i in range(epoch):
        out = []
        labels = []
        loss = []
        for batch in tqdm(dataloader):
            out = sc(batch[0], batch[1], batch[2])
            labels = batch[-1]
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if i % 5 ==0:
            out = out.argmax(dim = -1)
            acc = (out == labels).sum().item()/len(labels)
            logging.info(f"第{i}次，损失为：{loss}，准确率：{acc}")


train(load_data('chn_senti_corp-train.arrow', True), 100)
