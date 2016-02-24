from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.cluster import KMeans
import math

__author__ = 'IgorKarpov'


class GLCMPQSymmetricFinder(AbstractFinder):

    glcm_pq_cache = {}
    glcm_pq_codebooks = {}
    glcm_pq_precomputed_centroids_squared_distances = {}

    product_members_count = 0
    clusters_count = 0

    def __init__(self, data_source, descriptor_builder, product_members_count, clusters_count):
        super(GLCMPQSymmetricFinder, self).__init__(data_source, descriptor_builder)
        self.product_members_count = product_members_count
        self.clusters_count = clusters_count

    def learn(self, train_data_source, params_dict):

        images_count = train_data_source.get_count()

        flattened_descriptors = [None] * images_count
        for image_index in xrange(images_count):
            image = train_data_source.get_image(image_index)
            raw_descriptor = self.__descriptor_builder.build_descriptor(image)
            flattened_descriptors[image_index] = raw_descriptor

        train_set = numpy.array(flattened_descriptors)

        chunks = GLCMPQSymmetricFinder.split_array_of_arrays_by_columns(train_set, self.product_members_count)

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
            self.glcm_pq_precomputed_centroids_squared_distances[chunk_index] = \
                GLCMPQSymmetricFinder.compute_all_possible_squared_distances_combination(estimator.cluster_centers_)

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

        query_GLCM_descriptor = self.__descriptor_builder.build_descriptor(query_image)
        query_GLCM_PQ_descriptor = self.__build_glcm_pq_descriptor(query_GLCM_descriptor)

        for image_index in xrange(self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            image_GLCM_PQ_descriptor = self.glcm_pq_cache[image_file_name]
            distance = self.__calculate_pq_descriptors_distance(image_GLCM_PQ_descriptor, query_GLCM_PQ_descriptor)
            distances.append((image_index, distance))

        return distances

    def __calculate_pq_descriptors_distance(self, stored_descriptor, query_descriptor):
        """Both stored_descriptor and query_descriptor are PQ-descriptors"""
        distances_sum = 0
        for index in xrange(len(stored_descriptor)):
            stored_centroid_marker = stored_descriptor[index]
            query_centroid_marker = query_descriptor[index]

            a = stored_centroid_marker
            b = query_centroid_marker
            if a > b:
                a = query_centroid_marker
                b = stored_centroid_marker

            squared_distance = self.glcm_pq_precomputed_centroids_squared_distances[index][a][b]
            distances_sum += squared_distance

        distance = math.sqrt(distances_sum)
        return distance

    def __build_glcm_pq_descriptor(self, glcm_descriptor):
        flattened_glcm_descriptor = glcm_descriptor

        query_descriptor_chunks = GLCMPQSymmetricFinder.split_array_of_arrays_by_columns(
            numpy.array([flattened_glcm_descriptor]),
            self.product_members_count)

        pq_descriptor = [None]*len(query_descriptor_chunks)

        for index in xrange(0, len(query_descriptor_chunks)):
            glcm_chunk = query_descriptor_chunks[index]
            centroid = self.glcm_pq_codebooks[index]['estimator'].predict(glcm_chunk)[0]
            pq_descriptor[index] = centroid

        return pq_descriptor

    @staticmethod
    def split_array_of_arrays_by_columns(array_of_arrays, chunks_count):
        rows_count, cols_count = array_of_arrays.shape

        columns_in_chunk_count = (cols_count + chunks_count - 1) / chunks_count

        chunks = [None] * chunks_count

        for chunk_index in xrange(0, chunks_count):
            first_col_index = chunk_index * columns_in_chunk_count
            actual_chunk_size = min(columns_in_chunk_count, cols_count - first_col_index)
            chunks[chunk_index] = array_of_arrays[:,first_col_index:(first_col_index + actual_chunk_size)]

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
                distance = GLCMPQSymmetricFinder.__calculate_raw_vectors_distance(c1, c2)
                sq_distances[a][b] = distance**2

        return sq_distances