import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from modeling import PositionalEncoding, MultiheadAttention, LinearHeadAttention, TransformerEncoderLayer, TransformerEncoder

class TransformerModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.pos_encoder = PositionalEncoding(conf.model.d_model, conf.model.dropout)
        self.encoder = nn.Embedding(conf.model.vocab_size, conf.model.d_model)
        assert conf.model.attention in ['full', 'linear']
        if conf.model.attention == 'full':
            attn = MultiheadAttention(conf.model.d_model, conf.model.num_heads)
        elif conf.model.attention == 'linear':
            attn = LinearHeadAttention(conf.model.d_model, conf.model.num_heads, activation=lambda x: F.elu(x) + 1)
        encoder_layer = TransformerEncoderLayer(attn, conf.model.d_model, conf.model.dim_feedforward, conf.model.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, conf.model.num_layers)
        self.d_model = conf.model.d_model
        self.fc = nn.Linear(conf.model.d_model, conf.model.num_classes)
        self.conf = conf
        assert conf.metric in ['accuracy', 'mcc']
        if conf.metric == 'accuracy':
            self.metric = torchmetrics.Accuracy('binary')
        elif conf.metric == 'mcc':
            self.metric = torchmetrics.MatthewsCorrCoef('binary')
        

    def forward(self, x):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        output = self.fc(x.mean(1))
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch.input_ids)
        loss = F.cross_entropy(outputs, batch.labels)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch.input_ids)
        val_loss = F.cross_entropy(outputs, batch.labels)
        self.metric(outputs.argmax(1).view(-1), batch.labels)
        self.log("val_loss", val_loss)
        self.log(f"val_{self.conf.metric}", self.metric, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.conf.optimizer.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.conf.optimizer.T_max, eta_min = 0)
        return [optimizer], [lr_scheduler]
