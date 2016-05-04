from evaluator import Evaluator
from descriptor_builders.glcm_descriptor_builder import GLCMDescriptorBuilder
from descriptor_builders.lbp_histogram_descriptor_builder import LBPHistogramDescriptorBuilder
from descriptor_builders.gray_scale_histogram_descriptor_builder import GrayScaleHistogramDescriptorBuilder
from finders.glcm_rpq_symm_finder import GLCMRandomPQSymmetricFinder
from finders.glcm_finder import GLCMFinder
from finders.glcm_pca_finder import GLCMPCAFinder
from finders.glcm_pq_asymm_finder import GLCMPQAsymmetricFinder
from finders.glcm_pq_symm_finder import GLCMPQSymmetricFinder
from finders.glcm_rpq_asymm_finder import GLCMRandomPQAsymmetricFinder
from finders.glcm_rpq_symm_equality_finder import GLCMRandomPQSymmetricEqualityFinder
from finders.glcm_wta_finder import WTAFinder
import sys

__author__ = 'IgorKarpov'


def build_GLCMFinder(data_source, descriptor_builder, finder_parameters_set):
    finder = GLCMFinder(
        data_source,
        descriptor_builder)
    return finder

def build_GLCMPCAFinder(data_source, descriptor_builder, finder_parameters_set):
    finder = GLCMPCAFinder(
        data_source,
        descriptor_builder)
    return finder

def build_GLCMPQAsymmetricFinder(data_source, descriptor_builder, finder_parameters_set):

    product_members_count = finder_parameters_set[0]
    clusters_count = finder_parameters_set[1]

    finder = GLCMPQAsymmetricFinder(
        data_source,
        descriptor_builder,
        product_members_count,
        clusters_count)
    return finder

def build_GLCMPQSymmetricFinder(data_source, descriptor_builder, finder_parameters_set):

    product_members_count = finder_parameters_set[0]
    clusters_count = finder_parameters_set[1]

    finder = GLCMPQSymmetricFinder(
        data_source,
        descriptor_builder,
        product_members_count,
        clusters_count)
    return finder

def build_GLCMRandomPQAsymmetricFinder(data_source, descriptor_builder, finder_parameters_set):

    product_members_count = finder_parameters_set[0]
    product_member_size = finder_parameters_set[1]
    clusters_count = finder_parameters_set[2]

    finder = GLCMRandomPQAsymmetricFinder(
        data_source,
        descriptor_builder,
        product_members_count,
        product_member_size,
        clusters_count)
    return finder

def build_GLCMRandomPQSymmetricEqualityFinder(data_source, descriptor_builder, finder_parameters_set):

    product_members_count = finder_parameters_set[0]
    product_member_size = finder_parameters_set[1]
    clusters_count = finder_parameters_set[2]

    finder = GLCMRandomPQSymmetricEqualityFinder(
        data_source,
        descriptor_builder,
        product_members_count,
        product_member_size,
        clusters_count)
    return finder

def build_RandomPQSymmetricFinder(data_source, descriptor_builder, finder_parameters_set):
    finder = GLCMRandomPQSymmetricFinder(
        data_source,
        descriptor_builder,
        finder_parameters_set[0],
        finder_parameters_set[1],
        finder_parameters_set[2])
    return finder

def build_WTAFinder(data_source, descriptor_builder, finder_parameters_set):

    permutations_count = finder_parameters_set[0]
    active_permutation_length = finder_parameters_set[1]

    finder = WTAFinder(
        data_source,
        descriptor_builder,
        permutations_count,
        active_permutation_length)
    return finder

def build_GLCMDescriptorBuilder(image_depth, additional_params):
    distance = additional_params[0]
    builder = GLCMDescriptorBuilder(image_depth, distance)
    return builder

def build_LBPHistogramDescriptorBuilder(image_depth, additional_params):
    builder = LBPHistogramDescriptorBuilder()
    return builder

def build_GrayScaleHistogramDescriptorBuilder(image_depth, additional_params):
    builder = GrayScaleHistogramDescriptorBuilder(image_depth)
    return builder

# all these results computed with
# glcm_descriptor = greycomatrix(image, [5], [0], self.__image_depth, symmetric=True, normed=False).flatten()
#finder = GLCMFinder(data_source)
#finder = GLCMPCAFinder(data_source)
#finder = GLCMPQSymmetricFinder(data_source, 5, 200)
#finder = GLCMRandomPQSymmetricFinder(data_source, 3, 256*200, 500) # 89.5895895896 256bin
#finder = GLCMPQAsymmetricFinder(data_source, 5, 200) # 84.6846846847 256bin
#finder = GLCMRandomPQAsymmetricFinder(data_source, 3, 256*200, 500) # 87.5875875876 256bin
#finder = GLCMRandomPQSymmetricEqualityFinder(data_source, 3, 256*200, 500) # 64.8648648649 256bin
#finder = GLCMRandomPQSymmetricEqualityFinder(data_source, 40, 3000, 100) # 92.4924924925 256bin
#finder = GLCMRandomPQSymmetricFinder(data_source, 40, 3000, 100) # 85.8858858859 256bin
#finder = WTAFinder(data_source, 300, 1000) # 84.984984985 256bin
#finder = GLCMRandomPQSymmetricFinder(data_source, 3, 50, 200) # 60.2602602603 8bin
#finder = GLCMRandomPQSymmetricFinder(data_source, 10, 20, 100) # 58.958958959 8bin
#finder = GLCMRandomPQSymmetricFinder(data_source, 5, 30, 500) # 65.4654654655 8bin
#finder = GLCMRandomPQSymmetricFinder(data_source, 10, 20, 100) # 63.963963964 16bin
#finder = GLCMRandomPQSymmetricFinder(data_source, descriptor_builder, 40, 3, 100) # 64.964964965 8bin

def main():
    evaluator = Evaluator('brodatz_database_bd.gidx')


    # descriptor_pack =
    #   descriptor_name,
    #   descriptor_additional_parameters_names,
    #   descriptor_additional_parameters_sets,
    #   build_descriptor_builder
    # finder_pack =
    #   finder_name,
    #   finder_parameters_names,
    #   finder_parameters_sets,
    #   build_finder
    # def evaluate_accuracy(self,
    #                       image_depths_to_be_evaluated,
    #                       descriptors_packs,
    #                       finders_packs):

    descriptors_packs = [
        {
            'descriptor_name': 'GLCMDescriptor',
            'descriptor_additional_parameters_names': ['distance'],
            'descriptor_additional_parameters_sets': [[1], [2], [5], [10]],
            'build_descriptor_builder': build_GLCMDescriptorBuilder
        },
        {
            'descriptor_name': 'LBPHistogramDescriptorBuilder',
            'descriptor_additional_parameters_names': [],
            'descriptor_additional_parameters_sets': [[]],
            'build_descriptor_builder': build_LBPHistogramDescriptorBuilder
        },
        {
            'descriptor_name': 'GrayScaleHistogramDescriptorBuilder',
            'descriptor_additional_parameters_names': [],
            'descriptor_additional_parameters_sets': [[]],
            'build_descriptor_builder': build_GrayScaleHistogramDescriptorBuilder
        }
    ]

    rpq_finder_parameters_sets = [[10, 3, 10], [10, 5, 10], [10, 50, 10], [10, 200, 10],
                                       [200, 5, 10], [200, 50, 10], [200, 100, 10],
                                       [3, 5, 50], [3, 50, 50], [3, 100, 50], [3, 1000, 50],
                                       [5, 5, 50], [5, 50, 50], [5, 100, 50], [5, 1000, 50],
                                       [10, 5, 50], [10, 50, 50], [10, 100, 50], [10, 500, 50],
                                       [50, 5, 50], [50, 20, 50], [50, 50, 50], [50, 100, 50],
                                       [100, 5, 50], [100, 20, 50], [100, 50, 50], [100, 100, 50],
                                       [300, 5, 50], [300, 20, 50], [300, 50, 50], [300, 100, 50],
                                       [3, 5, 100], [3, 50, 100], [3, 100, 100], [3, 1000, 100], [3, 3000, 100],
                                       [5, 5, 100], [5, 50, 100], [5, 100, 100], [5, 1000, 100], [5, 3000, 100],
                                       [10, 5, 100], [10, 50, 100], [10, 100, 100], [10, 1000, 100], [10, 3000, 100],
                                       [50, 5, 100], [50, 50, 100], [50, 100, 100], [50, 1000, 100], [50, 3000, 100],
                                       [100, 5, 100], [100, 50, 100], [100, 100, 100], [100, 1000, 100], [100, 3000, 100],
                                       [3, 5, 200], [3, 50, 200],  [3, 500, 200],
                                       [5, 5, 200], [5, 50, 200], [5, 500, 200],
                                       [10, 5, 200], [10, 50, 200], [10, 500, 200]]


    finders_packs = [
        {
            'finder_name': 'GLCMFinder',
            'finder_parameters_names': [],
            'finder_parameters_sets': [[]],
            'build_finder': build_GLCMFinder
        },
        {
            'finder_name': 'GLCMPCAFinder',
            'finder_parameters_names': [],
            'finder_parameters_sets': [[]],
            'build_finder': build_GLCMPCAFinder
        },
        {
            'finder_name': 'GLCMPQAsymmetricFinder',
            'finder_parameters_names': ['product_members_count', 'clusters_count'],
            'finder_parameters_sets': [[10, 10], [200, 10],
                                       [3, 50], [5, 50], [10, 50], [50, 50], [100, 50], [300, 50],
                                       [3, 100], [5, 100], [10, 100], [50, 100], [100, 100],
                                       [3, 200], [5, 200], [10, 200]],
            'build_finder': build_GLCMPQAsymmetricFinder
        },
        {
            'finder_name': 'GLCMPQSymmetricFinder',
            'finder_parameters_names': ['product_members_count', 'clusters_count'],
            'finder_parameters_sets': [[10, 10], [200, 10],
                                       [3, 50], [5, 50], [10, 50], [50, 50], [100, 50], [300, 50],
                                       [3, 100], [5, 100], [10, 100], [50, 100], [100, 100],
                                       [3, 200], [5, 200], [10, 200]],
            'build_finder': build_GLCMPQSymmetricFinder
        },
        {
            'finder_name': 'GLCMRandomPQAsymmetricFinder',
            'finder_parameters_names': ['product_members_count', 'product_member_size', 'clusters_count'],
            'finder_parameters_sets': rpq_finder_parameters_sets,
            'build_finder': build_GLCMRandomPQAsymmetricFinder
        },
        {
            'finder_name': 'GLCMRandomPQSymmetricEqualityFinder',
            'finder_parameters_names': ['product_members_count', 'product_member_size', 'clusters_count'],
            'finder_parameters_sets': rpq_finder_parameters_sets,
            'build_finder': build_GLCMRandomPQSymmetricEqualityFinder
        },
        {
            'finder_name': 'RandomPQSymmetricFinder',
            'finder_parameters_names': ['product_members_count', 'product_member_size', 'clusters_count'],
            'finder_parameters_sets': rpq_finder_parameters_sets,
            'build_finder': build_RandomPQSymmetricFinder
        },
        {
            'finder_name': 'WTAFinder',
            'finder_parameters_names': ['permutations_count', 'active_permutation_length'],
            'finder_parameters_sets': [[5, 5], [10, 5],
                                       [5, 10], [10, 10], [50, 10], [100, 10], [1000, 10],
                                       [2, 50], [5, 50], [10, 50], [50, 50], [500, 50],
                                       [5, 200], [10, 200], [50, 200], [500, 200],
                                       [10, 1000], [300, 1000]],
            'build_finder': build_WTAFinder
        }
    ]

    chunks_count = 1
    chunk_index = 0

    if (len(sys.argv) > 1):
        chunks_count = int(float(sys.argv[1]))
        chunk_index = int(float(sys.argv[2]))

    evaluator.evaluate_accuracy(
       [8, 16, 256],
       descriptors_packs,
       finders_packs,
       chunks_count,
       chunk_index)


if __name__ == '__main__':
    main()
