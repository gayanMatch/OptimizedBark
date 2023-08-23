import torch
from glob import glob
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import get_data_loader
from pl_model import TrainModel
from ckpt import Bestckpt

class CustomCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.global_step % 5000 == 0:
            trainer.run_evaluation()


def main(cfg=None):
    exp_name = f'semantic_train'
    exp_cfg = cfg['semantic']
    exp_root_dir = f"{cfg['start_path']}/{exp_name}"
    # define model
    model = TrainModel(model_config=exp_cfg['model'], **exp_cfg['optim'])
    # define trainer
    trainer = pl.Trainer(
        default_root_dir=exp_root_dir,
        callbacks=ModelCheckpoint(**cfg['common']['ckpt']),
        max_steps=exp_cfg['optim']['max_iters'],
        gradient_clip_val=exp_cfg['optim']['gradient_clip'],
        **cfg['common']['trainer']
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # load best ckpt
    ckpt = Bestckpt(exp_root_dir)
    # define datas
    train_loader = get_data_loader('/datasets/train', int(cfg['semantic']['dataloader']['batch_size']))
    val_loader = get_data_loader('/datasets/val', int(cfg['semantic']['dataloader']['batch_size']), shuffle=False)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)


if __name__ == '__main__':
    cfg = yaml.safe_load(open('train.yaml'))
    torch.set_float32_matmul_precision('medium')
    main(cfg)
