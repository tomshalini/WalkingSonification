import torch
import torch.utils.data as data
from GaitSequenceDataset import GaitSequenceDataset

batch_size = 8

dataset = GaitSequenceDataset(root_dir = '../../KL_Study_HDF5_for_learning/data/',
                                longest_sequence = 85,
                                shortest_sequence = 55)


dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataloader))
