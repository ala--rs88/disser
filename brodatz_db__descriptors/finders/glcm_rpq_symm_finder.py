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

    def __init__(self, data_source, product_members_count, clusters_count):
        super(GLCMPQSymmetricFinder, self).__init__(data_source)
        self.product_members_count = product_members_count
        self.clusters_count = clusters_count