from evaluator import Evaluator
from descriptor_builders.glcm_descriptor_builder import GLCMDescriptorBuilder
from descriptor_builders.lbp_histogram_descriptor_builder import LBPHistogramDescriptorBuilder
from descriptor_builders.gray_scale_histogram_descriptor_builder import GrayScaleHistogramDescriptorBuilder
from finders.glcm_rpq_symm_finder import GLCMRandomPQSymmetricFinder

__author__ = 'IgorKarpov'


def build_random_PQ_symmetric_finder(data_source, descriptor_builder, finder_parameters_set):
    finder = GLCMRandomPQSymmetricFinder(
        data_source,
        descriptor_builder,
        finder_parameters_set[0],
        finder_parameters_set[1],
        finder_parameters_set[2])
    return finder


def main():
    evaluator = Evaluator('brodatz_database_bd.gidx')

    evaluator.evaluate_accuracy(
                          [16],
                          #lambda image_depth: GLCMDescriptorBuilder(image_depth),
                          #lambda image_depth: LBPHistogramDescriptorBuilder(),
                          lambda image_depth: GrayScaleHistogramDescriptorBuilder(image_depth),
                          [[5, 4, 100]],
                          build_random_PQ_symmetric_finder)

if __name__ == '__main__':
    main()
