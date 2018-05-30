# coding=utf-8
import numpy as np


class DataGenerator(object):
    def __init__(self, config):
        self.config = config
        # load data here
        self.features = np.ones((500, 64))
        self.labels = np.ones((500, 1))

    def next(self):
        idx = np.random.choice(1)
        yield {'features': self.features[idx], 'labels': self.labels[idx]}
