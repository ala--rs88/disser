from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.cluster import KMeans
import math
import random


__author__ = 'IgorKarpov'


class GLCMRandomPQAsymmetricFinder(AbstractFinder):

    glcm_pq_cache = {}
    glcm_pq_codebooks = {}
    glcm_pq_precomputed_centroids_squared_distances = {}
    random_product_members_first_columns_indexes = {} # for each product member stores index of the first column in the chunk

    product_members_count = 0
    product_member_size = 0
    clusters_count = 0

    def __init__(self, data_source, descriptor_builder, product_members_count, product_member_size, clusters_count):
        super(GLCMRandomPQAsymmetricFinder, self).__init__(data_source, descriptor_builder)
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
            raw_descriptor = GLCMRandomPQAsymmetricFinder.__build_glcm_descriptor(image)
            flattened_descriptors[image_index] = raw_descriptor.flatten()

        train_set = numpy.array(flattened_descriptors)

        chunks = GLCMRandomPQAsymmetricFinder.randomly_split_array_of_arrays_by_columns(
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

    def __calculate_distances(self, query_image):
        distances = []

        query_GLCM_descriptor = GLCMRandomPQAsymmetricFinder.__build_glcm_descriptor(query_image)
        flattened_query_GLCM_descriptor = query_GLCM_descriptor.flatten()
        chunked_query_GLCM_descriptor = GLCMRandomPQAsymmetricFinder.randomly_split_array_of_arrays_by_columns(
            numpy.array([flattened_query_GLCM_descriptor]),
            self.product_member_size,
            self.random_product_members_first_columns_indexes)

        squared_distances_cache = [{} for x in range(self.product_members_count)]

        for image_index in xrange(self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            image_PQ_descriptor = self.glcm_pq_cache[image_file_name]

            distance = self.__calculate_distance_between_pq_and_chunked_glcm_descriptors(
                image_PQ_descriptor,
                chunked_query_GLCM_descriptor,
                self.glcm_pq_codebooks,
                squared_distances_cache)
            distances.append((image_index, distance))

        return distances

    def __calculate_distance_between_pq_and_chunked_glcm_descriptors(
            self,
            stored_pq_descriptor,
            chunked_query_glcm_descriptor,
            glcm_pq_codebooks,
            squared_distances_cache):

        product_members_count = len(stored_pq_descriptor)

        distances_sum = 0
        for index in xrange(product_members_count):
            chunk_squared_distances_cache = squared_distances_cache[index]

            stored_centroid_label = stored_pq_descriptor[index]

            chunk_squared_distance = None
            if stored_centroid_label in chunk_squared_distances_cache:
                #print ('cache hit')
                chunk_squared_distance = chunk_squared_distances_cache[stored_centroid_label]
            else:
                #print ('cache miss')
                query_glcm_chunk = chunked_query_glcm_descriptor[index]
                stored_centroid_value = glcm_pq_codebooks[index]['estimator'].cluster_centers_[stored_centroid_label]
                chunk_distance = GLCMRandomPQAsymmetricFinder.__calculate_raw_vectors_distance(query_glcm_chunk, stored_centroid_value)
                chunk_squared_distance = chunk_distance**2

                chunk_squared_distances_cache[stored_centroid_label] = chunk_squared_distance

            distances_sum += chunk_squared_distance

        distance = math.sqrt(distances_sum)
        return distance

    @staticmethod
    def __build_glcm_descriptor(image):
        glcm_descriptor = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False)
        return glcm_descriptor

    @staticmethod
    def build_random_product_members_first_columns_indexes(total_size, product_members_count, product_member_size):

        random.seed(12345)

        product_members_first_columns_indexes = {}

        max_column_index = total_size - product_member_size
        for product_member_index in xrange(product_members_count):
            product_members_first_columns_indexes[product_member_index] = random.randint(0, max_column_index)

        return product_members_first_columns_indexes

    @staticmethod
    def randomly_split_array_of_arrays_by_columns(array_of_arrays, chunk_size, chunks_first_columns_indexes):
        chunks = [None] * len(chunks_first_columns_indexes)
        for chunk_index in xrange(0, len(chunks_first_columns_indexes)):
            first_col_index = chunks_first_columns_indexes[chunk_index]
            chunks[chunk_index] = array_of_arrays[:,first_col_index:(first_col_index + chunk_size)]

        return chunks

    @staticmethod
    def __calculate_raw_vectors_distance(vector1, vector2):
        #distance = cityblock(numpy.hstack(vector1.flatten()), numpy.hstack(vector2.flatten())) # -- L1
        distance = numpy.linalg.norm(vector1 - vector2) # -- L2
        return distance

    @staticmethod
    def compute_all_possible_squared_distances_combination(centroids_values):
        sq_distances = {}

        for a in xrange(0, len(centroids_values)):
            sq_distances[a] = {}
            for b in xrange(a, len(centroids_values)):
                c1 = centroids_values[a]
                c2 = centroids_values[b]
                distance = GLCMPQAsymmetricFinder.__calculate_raw_vectors_distance(c1, c2)
                sq_distances[a][b] = distance**2

        return sq_distances