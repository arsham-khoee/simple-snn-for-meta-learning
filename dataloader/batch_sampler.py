# coding=utf-8
import random
import numpy as np
import torch


class BatchSampler(object):
    '''
    BatchSampler: yield a batch of indexes at each iteration.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, batch_size):
        '''
        Initialize the BatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class
        - iterations: number of iterations (episodes) per epoch
        '''
        super(BatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.batch_size = batch_size

        self.classes, self.counts = np.unique(self.labels, return_counts=True)

        self.idxs = range(len(self.labels))
        self.label_tens = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.label_lens = np.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label)[0, 0]
            self.label_tens[label_idx, np.where(np.isnan(self.label_tens[label_idx]))[0][0]] = idx
            self.label_lens[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class + 1 # To get that extra sample
        cpi = self.classes_per_it
        num_samples = spc * cpi
        true_num_samples = (spc - 1) * cpi + 1
        for it in range(self.iterations):
            total_batch = np.array([])
            for _ in range(self.batch_size):
                batch = np.empty(num_samples)
                c_idxs = np.random.permutation(len(self.classes))[:cpi]
                for i, c in enumerate(self.classes[c_idxs]):
                    s = slice(i, i + num_samples, cpi)
                    label_idx = np.argwhere(self.classes == c)[0, 0]
                    if spc > self.label_lens[label_idx]:
                        raise AssertionError('More samples per class than exist in the dataset')
                    sample_idxs = np.random.permutation(self.label_lens[label_idx])[:spc]
                    batch[s] = self.label_tens[label_idx][sample_idxs]
                offset = random.randint(0, 4)
                batch = batch[offset:offset + true_num_samples]
                print(true_num_samples)
                print('****')
                print(batch)
                batch[:true_num_samples - 1] = batch[:true_num_samples - 1][np.random.permutation(true_num_samples - 1)]
                total_batch = np.append(total_batch, batch)
            yield total_batch.astype(int)

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
