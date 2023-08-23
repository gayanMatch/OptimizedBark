import os
import glob
import copy
import pickle
import random
import tqdm
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

START = 513
END = 513 + 768


class BarkDataset(Dataset):
    def __init__(self, path, batch_size):
        self.train_data = path
        self.batch_size = batch_size
        self.indices = []
        self.origin_data_files = []
        self.data_indices = []
        self.indices, self.origin_data_files, self.data_indices = pickle.load(open(f'dataset_{path.split("/")[-1]}.pkl', 'rb'))
        # for length in tqdm(range(START, END)):
        #     files = glob.glob(f"{path}/*_{length}.npz")
        #     indices = list(range(0, len(files)))
        #     random.shuffle(indices)
        #     self.data_indices.append(indices)
        #     self.origin_data_files.append(files)
        #     for i in range(len(files) // batch_size):
        #         self.indices.append(length - START)
        #
        # pickle.dump((self.indices, self.origin_data_files, self.data_indices), open(f'dataset_{path.split("/")[-1]}.pkl', 'wb'))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        length = self.indices[item]
        ret_data = []
        for i in range(self.batch_size):
            try:
                file_index = self.data_indices[length].pop(0)
            except:
                print()
            data = np.load(self.origin_data_files[length][file_index])
            if self.data_indices[length] == []:
                indices = list(range(0, len(self.origin_data_files[length])))
                random.shuffle(indices)
                self.data_indices[length] = indices
            ret_data.append(data)
        return ret_data


def collate_fn(batches):
    x_input_array = [x['x_input'] for x in batches[0]]
    logits_array = [x['logits'] for x in batches[0]]
    x_input = np.concatenate(x_input_array)
    logits = np.concatenate(logits_array)
    x_input = torch.from_numpy(x_input)
    logits = torch.from_numpy(logits)
    return x_input, logits


def get_data_loader(path, batch_size, shuffle=True):
    dataset = BarkDataset(path, batch_size)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=1, shuffle=shuffle)


if __name__ == '__main__':
    dataset = BarkDataset('/train', 2)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
    for batch in dataloader:
        print()
