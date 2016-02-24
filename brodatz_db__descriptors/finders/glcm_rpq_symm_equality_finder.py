from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.cluster import KMeans
import random

__author__ = 'IgorKarpov'


class GLCMRandomPQSymmetricEqualityFinder(AbstractFinder):

    glcm_pq_cache = {}
    glcm_pq_codebooks = {}
    random_product_members_first_columns_indexes = {} # for each product member stores index of the first column in the chunk

    product_members_count = 0
    product_member_size = 0
    clusters_count = 0

    def __init__(self, data_source, descriptor_builder, product_members_count, product_member_size, clusters_count):
        super(GLCMRandomPQSymmetricEqualityFinder, self).__init__(data_source, descriptor_builder)
        self.product_members_count = product_members_count
        self.product_member_size = product_member_size
        self.clusters_count = clusters_count

    def learn(self, train_data_source, params_dict):

        self.random_product_members_first_columns_indexes = self.build_random_product_members_first_columns_indexes(
            self.__descriptor_builder.get_descriptor_length(),
            self.product_members_count,
            self.product_member_size)

        images_count = train_data_source.get_count()

        flattened_descriptors = [None] * images_count
        for image_index in xrange(images_count):
            image = train_data_source.get_image(image_index)
            raw_descriptor = GLCMRandomPQSymmetricEqualityFinder.__build_glcm_descriptor(image)
            flattened_descriptors[image_index] = raw_descriptor.flatten()

        train_set = numpy.array(flattened_descriptors)

        chunks = GLCMRandomPQSymmetricEqualityFinder.randomly_split_array_of_arrays_by_columns(
            train_set,
            self.product_member_size,
            self.random_product_members_first_columns_indexes)

        for image_index in xrange(images_count):
            image_name = train_data_source.get_image_file_name(image_index)
            self.glcm_pq_cache[image_name] = [None] * self.product_members_count

        for chunk_index in xrange(len(chunks)):
            chunk = chunks[chunk_index]
            estimator = KMeans(n_clusters=self.clusters_count)
            estimator.fit(chunk)

            self.glcm_pq_codebooks[chunk_index] = {
                'clusters_labels': estimator.labels_,
                'estimator': estimator
            }

            for image_index in xrange(images_count):
                image_name = train_data_source.get_image_file_name(image_index)
                image_descriptor = self.glcm_pq_cache[image_name]
                image_descriptor[chunk_index] = estimator.labels_[image_index]

            print 'chunk calculated: ' + repr(chunk_index)


    def find_top_matches(self, query_image, top_count):
        distances = self.__calculate_distances(query_image)
        distances.sort(key=lambda tup: tup[1])
        top_matching_images_ids = [x[0] for x in distances[:5]]
        return top_matching_images_ids

    def build_random_product_members_first_columns_indexes(self, total_size, product_members_count, product_member_size):

        random.seed(12345)

        product_members_first_columns_indexes = {}

        max_column_index = total_size - product_member_size
        for product_member_index in xrange(product_members_count):
            product_members_first_columns_indexes[product_member_index] = random.randint(0, max_column_index)

        return product_members_first_columns_indexes

    def __calculate_distances(self, query_image):
        distances = []

        query_GLCM_descriptor = self.__build_glcm_descriptor(query_image)
        query_GLCM_PQ_descriptor = self.__build_glcm_pq_descriptor(query_GLCM_descriptor)

        for image_index in xrange(self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            image_GLCM_PQ_descriptor = self.glcm_pq_cache[image_file_name]
            distance = self.__calculate_pq_descriptors_distance_based_on_equality_of_product_members(image_GLCM_PQ_descriptor, query_GLCM_PQ_descriptor)
            distances.append((image_index, distance))

        return distances

    def __calculate_pq_descriptors_distance_based_on_equality_of_product_members(self, stored_descriptor, query_descriptor):
        """Both stored_descriptor and query_descriptor are PQ-descriptors"""
        number_of_nonmatching_product_members = 0
        for index in xrange(len(stored_descriptor)):
            stored_centroid_marker = stored_descriptor[index]
            query_centroid_marker = query_descriptor[index]

            if stored_centroid_marker != query_centroid_marker:
                number_of_nonmatching_product_members += 1

        distance = number_of_nonmatching_product_members
        return distance

    @staticmethod
    def __build_glcm_descriptor(image):
        glcm_descriptor = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False)
        return glcm_descriptor

    def __build_glcm_pq_descriptor(self, glcm_descriptor):
        flattened_glcm_descriptor = glcm_descriptor.flatten()

        query_descriptor_chunks = GLCMRandomPQSymmetricEqualityFinder.randomly_split_array_of_arrays_by_columns(
            numpy.array([flattened_glcm_descriptor]),
            self.product_member_size,
            self.random_product_members_first_columns_indexes)

        pq_descriptor = [None]*len(query_descriptor_chunks)

        for index in xrange(0, len(query_descriptor_chunks)):
            glcm_chunk = query_descriptor_chunks[index]
            centroid = self.glcm_pq_codebooks[index]['estimator'].predict(glcm_chunk)[0]
            pq_descriptor[index] = centroid

        return pq_descriptor

    @staticmethod
    def randomly_split_array_of_arrays_by_columns(array_of_arrays, chunk_size, chunks_first_columns_indexes):
        chunks = [None] * len(chunks_first_columns_indexes)
        for chunk_index in xrange(0, len(chunks_first_columns_indexes)):
            first_col_index = chunks_first_columns_indexes[chunk_index]
            chunks[chunk_index] = array_of_arrays[:,first_col_index:(first_col_index + chunk_size)]

        return chunks