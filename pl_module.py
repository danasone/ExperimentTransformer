import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import bitsandbytes as bnb
import pytorch_lightning as pl
from modeling import PositionalEncoding, MultiheadAttention, LinearHeadAttention, TransformerEncoderLayer, TransformerEncoder
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

class TransformerModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.pos_encoder = PositionalEncoding(conf.model.d_model, conf.model.dropout)
        if conf.precision == 8:
            self.encoder = bnb.nn.StableEmbedding(conf.model.vocab_size, conf.model.d_model)
        else:
            self.encoder = nn.Embedding(conf.model.vocab_size, conf.model.d_model)
        assert conf.model.attention in ['full', 'linear']
        if conf.model.attention == 'full':
            attn = MultiheadAttention(conf.model.d_model, conf.model.num_heads, conf.precision)
        elif conf.model.attention == 'linear':
            attn = LinearHeadAttention(conf.model.d_model, conf.model.num_heads, conf.precision, activation=lambda x: F.elu(x) + 1)
        encoder_layer = TransformerEncoderLayer(attn, conf.model.d_model, conf.model.dim_feedforward, conf.model.dropout, conf.precision)
        self.transformer_encoder = TransformerEncoder(encoder_layer, conf.model.num_layers)
        self.d_model = conf.model.d_model
        if conf.precision == 8:
            self.fc = bnb.nn.Linear8bitLt(conf.model.d_model, conf.model.num_classes, bias=True, has_fp16_weights=True, threshold=6.0)
        else:
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
        if self.conf.precision == 8:
            optimizer = bnb.optim.Adam8bit(self.parameters(), lr=self.conf.optimizer.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.conf.optimizer.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.conf.optimizer.T_max, eta_min = 0)
        return [optimizer], [lr_scheduler]
    
    
def replace_8bit_linear(model, threshold=6.0, module_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_not_convert)

        if isinstance(module, nn.Linear) and name != module_to_not_convert:
            with init_empty_weights():
                model._modules[name] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=True,
                    threshold=threshold,
                )
        if isinstance(module, nn.Embedding) and name != module_to_not_convert:
            with init_empty_weights():
                model._modules[name] = bnb.nn.StableEmbedding(
                    module.num_embeddings,
                    module.embedding_dim
                )        
    return model
    
    
class DebertaModel(TransformerModel):
    def __init__(self, conf, path):
        super().__init__(conf)
        deberta_config = DebertaV2Config.from_pretrained(path)
        deberta_config.num_labels = conf.model.num_classes
        self.encoder = DebertaV2ForSequenceClassification(deberta_config)
        if conf.precision == 8:
            self.encoder = replace_8bit_linear(self.encoder)
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin')))
        self.conf = conf
        assert conf.metric in ['accuracy', 'mcc']
        if conf.metric == 'accuracy':
            self.metric = torchmetrics.Accuracy('binary')
        elif conf.metric == 'mcc':
            self.metric = torchmetrics.MatthewsCorrCoef('binary')
            
    def forward(self, x):
        x = self.encoder(x)
        return x.logits
        
            
