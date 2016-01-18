__author__ = 'IgorKarpov'

from abc import ABCMeta, abstractmethod


class AbstractFinder:
    __metaclass__ = ABCMeta

    data_source = None

    def __init__(self, data_source):
        self.data_source = data_source

    @abstractmethod
    def learn(self, train_data_source, params_dict):
        pass

    @abstractmethod
    def find_top_matches(self, query_image, top_count):
        pass