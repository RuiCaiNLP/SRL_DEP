import itertools
import functools
import random

def simple_reader(data, batch_size):
    print([iter(enumerate(data))])
    args = [iter(enumerate(data))] * batch_size

    for batch in itertools.izip_longest(*args, fillvalue=None):
        yield [s for s in batch if s is not None]


class Corpus(object):
    def __init__(self, parser, batch_size, path, reader=simple_reader):
        self.parser = parser
        self.path = path
        self.bath_size = batch_size
        self._size = None
        self.reader = reader

    def batches(self):
        with open(self.path, 'r') as data:
            for batch in self.reader(data, self.bath_size):
                batch = [(id, self.parser(record[:-1])) for (id, record) in batch]
                yield batch

    def get_batch_size(self):
        return self.get_batch_size()
