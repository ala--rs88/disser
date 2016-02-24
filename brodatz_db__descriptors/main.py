from evaluator import Evaluator
from descriptor_builders.glcm_descriptor_builder import GLCMDescriptorBuilder
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
                          [8, 16],
                          lambda image_depth: GLCMDescriptorBuilder(image_depth),
                          [[40, 3, 100], [10, 20, 100]],
                          build_random_PQ_symmetric_finder)

if __name__ == '__main__':
    main()
