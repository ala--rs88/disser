from abstract_descriptor_builder import AbstractDescriptorBuilder
from skimage.feature import local_binary_pattern
from numpy import histogram

__author__ = 'IgorKarpov'


class LBPHistogramDescriptorBuilder(AbstractDescriptorBuilder):

    __descriptor_length = -1
    __neighbours_count = -1
    __possible_lbp_values_count = -1

    def __init__(self):
        self.__neighbours_count = 8
        self.__possible_lbp_values_count = 2**self.__neighbours_count
        self.__descriptor_length = self.__possible_lbp_values_count

    def get_descriptor_length(self):
        return self.__descriptor_length

    def build_descriptor(self, image):
        lbp_matrix = local_binary_pattern(image, self.__neighbours_count, 1, 'ror')
        lbp_hist, bin_edges = histogram(lbp_matrix, bins=range(self.__possible_lbp_values_count + 1))
        return lbp_hist
