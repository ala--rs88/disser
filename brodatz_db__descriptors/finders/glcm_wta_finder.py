from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.cluster import KMeans
import random
import math
import itertools

__author__ = 'IgorKarpov'


class WTAFinder(AbstractFinder):

    FLATTENED_GLCM_DESCRIPTOR_SIZE = 256 * 256

    wta_hashes_cache = {}
    random_permutations_indexes = []  # each permutation is represented by an array of indexes of cells in descriptor

    active_permutation_length = 0
    permutations_count = 0

    def __init__(self, data_source, permutations_count, active_permutation_length):
        super(WTAFinder, self).__init__(data_source)
        self.active_permutation_length = active_permutation_length
        self.permutations_count = permutations_count

    def learn(self, train_data_source, params_dict):

        self.random_permutations_indexes = WTAFinder.__build_random_permutations_indexes(
            self.FLATTENED_GLCM_DESCRIPTOR_SIZE,
            self.active_permutation_length,
            self.permutations_count)

        images_count = train_data_source.get_count()

        for image_index in xrange(images_count):
            image_name = train_data_source.get_image_file_name(image_index)
            image = train_data_source.get_image(image_index)
            self.wta_hashes_cache[image_name] = WTAFinder.__build_wta_hash(image,
                                                                           self.random_permutations_indexes)
        print('WTA Hashes computed')

    def find_top_matches(self, query_image, top_count):
        distances = self.__calculate_distances(query_image)
        distances.sort(key=lambda tup: tup[1])
        top_matching_images_ids = [x[0] for x in distances[:5]]
        return top_matching_images_ids

    def __calculate_distances(self, query_image):
        distances = []

        query_hash = WTAFinder.__build_wta_hash(query_image,
                                                self.random_permutations_indexes)

        for image_index in xrange(self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            stored_hash = self.wta_hashes_cache[image_file_name]
            distance = self.__calculate_distance(query_hash, stored_hash)
            distances.append((image_index, distance))

        return distances

    @staticmethod
    def __calculate_distance(vector1, vector2):
        #distance = cityblock(numpy.hstack(vector1.flatten()), numpy.hstack(vector2.flatten())) # -- L1
        distance = numpy.linalg.norm(vector1 - vector2) # -- L2
        return distance

    @staticmethod
    def __build_glcm_descriptor(image):
        glcm_descriptor = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False).flatten()
        return glcm_descriptor

    @staticmethod
    def __build_wta_hash(image, random_permutations_indexes):
        descriptor = WTAFinder.__build_glcm_descriptor(image)
        random_active_permutations = []

        for permutation_indexes in random_permutations_indexes:
            permutation = [descriptor[i] for i in permutation_indexes]
            random_active_permutations.append(permutation)

        hash_len = len(random_active_permutations)

        wta_hash = [None] * hash_len
        for index in xrange(hash_len):
            active_permutation_part = random_active_permutations[index]
            max_element_index = active_permutation_part.index(max(active_permutation_part))
            wta_hash[index] = max_element_index

        numpy_wta_hash = numpy.array(wta_hash)

        return numpy_wta_hash

    @staticmethod
    def __build_random_permutations_indexes(total_permutation_length, active_permutation_length, permutations_count):
        indexes = range(total_permutation_length)
        permutations_indexes = []

        numpy.random.seed(12345)
        for i in xrange(permutations_count):
            permutations_indexes.append(numpy.random.permutation(indexes)[:active_permutation_length])

        return permutations_indexes
