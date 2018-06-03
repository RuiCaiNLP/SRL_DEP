from __future__ import print_function
import functools
import numpy as np
import sys



def log(*args, **kwargs):
    print(file=sys.stderr, *args, **kwargs)


def mask_batch(batch):
    max_len = len(max(batch, key=len))
    mask = np.zeros((len(batch), max_len))
    padded = np.zeros((len(batch), max_len))
    for i in range(len(batch)):
        mask[i, :len(batch[i])] = 1
        for j in range(len(batch[i])):
            padded[i, j] = batch[i][j]

    #return padded.astype('int32'), mask.astype(T.config.floatX)
    return padded.astype('int64'), mask.astype('int64')


def parse_word_embeddings(embeddings):

    res = []

    for line in open(embeddings, 'r'):
        emb = map(float, line.strip().split()[1:])
        res.append(list(emb))


    return np.array(res, dtype='float32')