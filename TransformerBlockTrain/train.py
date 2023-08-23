import torch
import yaml
from model import GPT, GPTConfig
from torch.utils.data import DataLoader
from datasets import BarkDataset, collate_fn

TRAIN_DIR = '/train'
MODEL_DIR = '/model'
CONFIG_PATH = 'block_config.yaml'
BATCH_SIZE = 2

def build_model():
    config = GPTConfig(**yaml.safe_load(open(CONFIG_PATH)))
    return GPT(config)

def load_model():
    pass

def save_model(model, optimizer, step):
    pass


def main():
    dataset = BarkDataset(TRAIN_DIR, BATCH_SIZE)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=1)
    model = build_model()
    for x_input, target in dataloader:
        logits, _ = model(x_input, merge_context=True, use_cache=False)
        print()

if __name__ == '__main__':
    main()
