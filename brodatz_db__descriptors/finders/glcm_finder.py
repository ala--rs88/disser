from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy

__author__ = 'IgorKarpov'


class GLCMFinder(AbstractFinder):

    __descriptors_cache = {}

    def __init__(self, data_source, descriptor_builder):
        super(GLCMFinder, self).__init__(data_source, descriptor_builder)

    def learn(self, train_data_source, params_dict):
        self.__descriptors_cache = self.__build_descriptors_cache(train_data_source)

    def find_top_matches(self, query_image, top_count):
        distances = self.__calculate_distances(query_image)
        distances.sort(key=lambda tup: tup[1])
        top_matching_images_ids = [x[0] for x in distances[:5]]
        return top_matching_images_ids

    def __build_descriptors_cache(self, data_source):
        descriptors_cache = {}
        for image_index in xrange(0, data_source.get_count()):
            image = self.data_source.get_image(image_index)
            descriptor = self.descriptor_builder.build_descriptor(image)
            image_file_name = self.data_source.get_image_file_name(image_index)
            descriptors_cache[image_file_name] = descriptor
        return descriptors_cache

    def __calculate_distances(self, query_image):
        distances = []

        query_image_descriptor = self.descriptor_builder.build_descriptor(query_image)

        for image_index in xrange(0, self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            image_descriptor = self.__descriptors_cache[image_file_name]
            distance = GLCMFinder.__calculate_distance(image_descriptor, query_image_descriptor)
            distances.append((image_index, distance))

        return distances

    @staticmethod
    def __calculate_distance(vector1, vector2):
        #distance = cityblock(numpy.hstack(vector1.flatten()), numpy.hstack(vector2.flatten())) # -- L1
        distance = numpy.linalg.norm(vector1 - vector2) # -- L2
        return distance
