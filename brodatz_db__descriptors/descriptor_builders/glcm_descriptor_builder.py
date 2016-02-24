from abstract_descriptor_builder import AbstractDescriptorBuilder
from skimage.feature import greycomatrix

__author__ = 'IgorKarpov'


class GLCMDescriptorBuilder(AbstractDescriptorBuilder):

    __image_depth = -1
    __descriptor_length = 0

    def __init__(self, image_depth):
        self.__image_depth = image_depth
        self.__descriptor_length = image_depth ** 2

    def get_descriptor_length(self):
        return self.__descriptor_length

    def build_descriptor(self, image):
        glcm_descriptor = greycomatrix(image, [5], [0], self.__image_depth, symmetric=True, normed=False).flatten()
        return glcm_descriptor