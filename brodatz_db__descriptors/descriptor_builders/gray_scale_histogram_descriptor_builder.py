from abstract_descriptor_builder import AbstractDescriptorBuilder
from skimage.feature import local_binary_pattern
from numpy import histogram

__author__ = 'IgorKarpov'


class GrayScaleHistogramDescriptorBuilder(AbstractDescriptorBuilder):

    __descriptor_length = -1
    __image_depth = -1

    def __init__(self, image_depth):
        self.__image_depth = image_depth
        self.__descriptor_length = image_depth

    def get_descriptor_length(self):
        return self.__descriptor_length

    def build_descriptor(self, image):
        gray_scale_hist, bin_edges = histogram(image, bins=range(self.__image_depth + 1))
        return gray_scale_hist
