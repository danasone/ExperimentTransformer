import os
from argparse import ArgumentParser
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from dataset import GLUEDataModule
from pl_module import TransformerModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/baseline.yaml')
    args = parser.parse_args()
    
    conf = OmegaConf.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)
    conf.model.vocab_size = tokenizer.vocab_size
    data_module = GLUEDataModule(conf, tokenizer)
    data_module.setup('fit')
    if conf.optimizer.T_max == 0:
        conf.optimizer.T_max = conf.num_epoch * len(data_module.train_dataloader())
    model = TransformerModel(conf)
    
    checkpoint_callback = ModelCheckpoint(monitor=f"val_{conf.metric}", mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tqdm_progress = TQDMProgressBar(refresh_rate=conf.trainer.log_steps)
    logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")
    trainer = pl.Trainer(max_epochs=conf.num_epoch,
                        accelerator=conf.trainer.accelerator,
                        accumulate_grad_batches=conf.trainer.accumulate,
                        callbacks=[checkpoint_callback, lr_monitor, tqdm_progress],
                        logger=logger,
                        log_every_n_steps=conf.trainer.log_steps)
    trainer.fit(model, data_module)
