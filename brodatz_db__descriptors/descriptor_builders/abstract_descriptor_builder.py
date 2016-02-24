from abc import ABCMeta, abstractmethod

__author__ = 'IgorKarpov'


class AbstractDescriptorBuilder:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_descriptor_length(self):
        pass

    @abstractmethod
    def build_descriptor(self, image):
        pass