from evaluator import Evaluator
from descriptor_builders.glcm_descriptor_builder import GLCMDescriptorBuilder
from descriptor_builders.lbp_histogram_descriptor_builder import LBPHistogramDescriptorBuilder
from descriptor_builders.gray_scale_histogram_descriptor_builder import GrayScaleHistogramDescriptorBuilder
from finders.glcm_rpq_symm_finder import GLCMRandomPQSymmetricFinder

__author__ = 'IgorKarpov'


def build_RandomPQSymmetricFinder(data_source, descriptor_builder, finder_parameters_set):
    finder = GLCMRandomPQSymmetricFinder(
        data_source,
        descriptor_builder,
        finder_parameters_set[0],
        finder_parameters_set[1],
        finder_parameters_set[2])
    return finder

def build_GLCMDescriptorBuilder(image_depth, additional_params):
    builder = GLCMDescriptorBuilder(image_depth)
    return builder


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

    descriptors_packs = [{
        'descriptor_name': 'GLCMDescriptor',
        'descriptor_additional_parameters_names': [],
        'descriptor_additional_parameters_sets': [[]],
        'build_descriptor_builder': build_GLCMDescriptorBuilder
    }]

    finders_packs = [{
        'finder_name': 'RandomPQSymmetricFinder',
        'finder_parameters_names': ['product_members_count', 'product_member_size', 'clusters_count'],
        'finder_parameters_sets': [[5, 4, 10], [3, 2, 10]],
        'build_finder': build_RandomPQSymmetricFinder
    }]

    evaluator.evaluate_accuracy(
        [8, 16],
        descriptors_packs,
        finders_packs)


if __name__ == '__main__':
    main()
