from abstract_finder import AbstractFinder
from skimage.feature import greycomatrix
import numpy
from sklearn.decomposition import PCA

__author__ = 'IgorKarpov'


class GLCMPCAFinder(AbstractFinder):

    __descriptors_cache = {}
    pca = None

    def __init__(self, data_source, descriptor_builder):
        super(GLCMPCAFinder, self).__init__(data_source, descriptor_builder)

    def learn(self, train_data_source, params_dict):
        flattened_descriptors = [None] * train_data_source.get_count()
        for image_index in xrange(train_data_source.get_count()):
            image = train_data_source.get_image(image_index)
            descriptor = self.__descriptor_builder.build_descriptor(image)
            flattened_descriptors[image_index] = descriptor

        PCA_train_set = numpy.array(flattened_descriptors)

        self.pca = PCA(n_components=0.9999)
        transformed_train_set = self.pca.fit_transform(PCA_train_set)

        for image_index in xrange(train_data_source.get_count()):
            image_name = train_data_source.get_image_file_name(image_index)
            self.__descriptors_cache[image_name] = transformed_train_set[image_index]


    def find_top_matches(self, query_image, top_count):
        distances = self.__calculate_distances(query_image)
        distances.sort(key=lambda tup: tup[1])
        top_matching_images_ids = [x[0] for x in distances[:5]]
        return top_matching_images_ids

    def __calculate_distances(self, query_image):
        distances = []

        query_image_glcm_descriptor = self.__descriptor_builder.build_descriptor(query_image)
        query_image_glcm_pca_descriptor = self.pca.transform(query_image_glcm_descriptor)

        for image_index in xrange(0, self.data_source.get_count()):
            image_file_name = self.data_source.get_image_file_name(image_index)
            image_descriptor = self.__descriptors_cache[image_file_name]
            distance = GLCMPCAFinder.__calculate_distance(image_descriptor, query_image_glcm_pca_descriptor)
            distances.append((image_index, distance))

        return distances

    @staticmethod
    def __calculate_distance(vector1, vector2):
        #distance = cityblock(numpy.hstack(vector1.flatten()), numpy.hstack(vector2.flatten())) # -- L1
        distance = numpy.linalg.norm(vector1 - vector2) # -- L2
        return distance