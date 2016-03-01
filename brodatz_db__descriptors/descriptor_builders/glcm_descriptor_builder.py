from abstract_descriptor_builder import AbstractDescriptorBuilder
from skimage.feature import greycomatrix
from numpy import pi

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
        glcms = greycomatrix(image,
                             [2],
                             [0, (1./4)*pi, (1./2)*pi, (3./4)*pi],
                             self.__image_depth,
                             symmetric=False,
                             normed=False)
        glcm_descriptor = (glcms[:, :, 0, 0] + glcms[:, :, 0, 1] + glcms[:, :, 0, 2] + glcms[:, :, 0, 3]).flatten()
        return glcm_descriptor