from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.cluster import KMeans
import math

__author__ = 'IgorKarpov'


class GLCMPQAsymmetricFinder(AbstractFinder):

    glcm_pq_cache = {}
    glcm_pq_codebooks = {}
    glcm_pq_precomputed_centroids_squared_distances = {}

    product_members_count = 0
    clusters_count = 0

    def __init__(self, data_source, descriptor_builder, product_members_count, clusters_count):
        super(GLCMPQAsymmetricFinder, self).__init__(data_source, descriptor_builder)
        self.product_members_count = product_members_count
        self.clusters_count = clusters_count

    def learn(self, train_data_source, params_dict):

        images_count = train_data_source.get_count()

        flattened_descriptors = [None] * images_count
        for image_index in xrange(images_count):
            image = train_data_source.get_image(image_index)
            descriptor = self.descriptor_builder.build_descriptor(image)
            flattened_descriptors[image_index] = descriptor

        train_set = numpy.array(flattened_descriptors)

        chunks = GLCMPQAsymmetricFinder.split_array_of_arrays_by_columns(train_set, self.product_members_count)

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

        flattened_query_GLCM_descriptor = self.descriptor_builder.build_descriptor(query_image)
        chunked_query_GLCM_descriptor = GLCMPQAsymmetricFinder.split_array_of_arrays_by_columns(
            numpy.array([flattened_query_GLCM_descriptor]),
            self.product_members_count)

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
                chunk_distance = GLCMPQAsymmetricFinder.__calculate_raw_vectors_distance(query_glcm_chunk, stored_centroid_value)
                chunk_squared_distance = chunk_distance**2

                chunk_squared_distances_cache[stored_centroid_label] = chunk_squared_distance

            distances_sum += chunk_squared_distance

        distance = math.sqrt(distances_sum)
        return distance

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
                distance = GLCMPQAsymmetricFinder.__calculate_raw_vectors_distance(c1, c2)
                sq_distances[a][b] = distance**2

        return sq_distances