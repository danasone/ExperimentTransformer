import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class GLUEDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.tokenizer.encode(self.data[idx]['sentence']), 'labels': torch.tensor(self.data[idx]['label'])}
      

class GLUEDataModule(pl.LightningDataModule):
    def __init__(self, conf, tokenizer):
        super().__init__()
        self.dataset = load_dataset("glue", conf.task)
        self.conf = conf
        self.tokenizer = tokenizer
        self.collator = DataCollatorWithPadding(self.tokenizer, max_length=conf.max_length, padding='max_length')

    def setup(self, stage: str):
        self.train_ds = GLUEDataset(self.dataset['train'], self.tokenizer)
        self.valid_ds = GLUEDataset(self.dataset['validation'], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.conf.batch_size, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.conf.batch_size, collate_fn=self.collator)
