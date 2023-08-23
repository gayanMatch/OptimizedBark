import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup

from model import GPT, GPTConfig


class TrainModel(pl.LightningModule):
    def __init__(
            self,
            model_config,
            lr,
            min_lr,
            weight_decay,
            warmup_iters,
            max_iters,
            lr_strategy='constant',
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._create_model()

    def _create_model(self):
        gptconf = GPTConfig(**self.hparams.model_config)
        self.model = GPT(gptconf)
        weight = torch.load('word_embedding.pth')
        self.model.transformer.wte.load_state_dict(weight)

    def forward(self, x, ) -> torch.Tensor:
        logits, _ = self.model(x, use_cache=False, merge_context=True)
        return logits

    @torch.no_grad()
    def infer(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.optimizers().param_groups[0]['lr'] < float(self.hparams.min_lr) and self.lr_schedulers().last_epoch > int(self.hparams.warmup_iters):
            self.optimizers().param_groups[0]['lr'] = float(self.hparams.min_lr)
            self.lr_schedulers().last_epoch -= 1

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if self.hparams.lr_strategy == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
            )
        elif self.hparams.lr_strategy == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
                num_training_steps=self.hparams.max_iters,
            )
        elif self.hparams.lr_strategy == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_iters,
                num_training_steps=self.hparams.max_iters,
            )
        else:
            raise NotImplementedError

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch[0])
        loss = self.criterion(F.log_softmax(logits[:, 0, :], dim=1), F.log_softmax(batch[1][:, 0, :], dim=1))
        # top1_acc, top10_acc = self.get_top10_acc(logits[:1, :-1, :], batch[1][:1, 1:])
        if (self.global_step + 1) % 5000 == 0:  # Save after every 10000 steps
            torch.save({'model': self.state_dict()},
                       f'models/checkpoint_at_step_{self.global_step}.pt')
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_1acc", top1_acc)
        # self.log("train_10acc", top10_acc)

        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        self.log("lr", self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch[0])
        loss = self.criterion(F.log_softmax(logits[:, 0, :], dim=1), F.log_softmax(batch[1][:, 0, :], dim=1))

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_top10_acc(self, logits, labels):
        with torch.no_grad():
            top10 = logits.topk(dim=-1, k=10).indices
            top1 = logits.topk(dim=-1, k=1).indices
            label = labels.unsqueeze(-1)

            total_item = (label != -100).sum()
            top10_acc = ((top10 == label).any(-1, keepdim=True) & (label != -100)).sum() / total_item
            top1_acc = ((top1 == label).any(-1, keepdim=True) & (label != -100)).sum() / total_item
            top10_acc, top1_acc = top10_acc.item(), top1_acc.item()

        return top1_acc, top10_acc