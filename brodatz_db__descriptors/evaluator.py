from data_source import DataSource
#from finders.glcm_finder import GLCMFinder
from finders.glcm_pca_finder import GLCMPCAFinder
from finders.glcm_pq_symm_finder import GLCMPQSymmetricFinder

from finders.glcm_pq_asymm_finder import GLCMPQAsymmetricFinder
from finders.glcm_rpq_asymm_finder import GLCMRandomPQAsymmetricFinder
from finders.glcm_rpq_symm_equality_finder import GLCMRandomPQSymmetricEqualityFinder
from finders.glcm_wta_finder import WTAFinder
from knn_classifier import kNNClassifier
import os
import numpy
import random

__author__ = 'IgorKarpov'

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

class Evaluator:

    __files_path = None
    __file_names = None

    def __init__(self, files_path):
        self.__files_path = files_path
        self.__file_names = Evaluator.__get_images_names(files_path, 'png')
        # self.__file_names = random.sample(self.__file_names, 100)
        #print self.__file_names
        print ''

    def evaluate_accuracy(self,
                          image_depths_to_be_evaluated,
                          build_descriptor_builder,
                          finder_parameters_sets_to_be_evaluated,
                          build_finder):

        # TODO: use using
        # TODO: create new before begin, then open-append-close in cycle
        f = open(os.path.join("evaluation_results", "test_results"), 'w')

        for image_depth in image_depths_to_be_evaluated:
            for finder_parameters_set in finder_parameters_sets_to_be_evaluated:
                Evaluator.reset_environment()
                accuracy = Evaluator.evaluate_set_accuracy(self.__files_path,
                                                           self.__file_names,
                                                           image_depth,
                                                           build_descriptor_builder,
                                                           finder_parameters_set,
                                                           build_finder)

                f.write('SET EVALUATED:\n')
                f.write('accuracy = ' + repr(accuracy) + '\n')
                f.write('depth = ' + repr(image_depth) + '\n')
                f.write('finder_parameters_set = ' + repr(finder_parameters_set) + '\n\n')

                print('\n\nSET EVALUATED:')
                print('accuracy = ' + repr(accuracy))
                print('depth = ' + repr(image_depth))
                print('finder_parameters_set = ' + repr(finder_parameters_set))
                print('\n\n')

        f.close()

    @staticmethod
    def evaluate_set_accuracy(files_path,
                              file_names,
                              image_depth,
                              build_descriptor_builder,
                              finder_parameters_set,
                              build_finder):
        data_source = DataSource(files_path, file_names, image_depth)
        descriptor_builder = build_descriptor_builder(image_depth)
        finder = build_finder(data_source, descriptor_builder, finder_parameters_set)
        classifier = kNNClassifier(5, finder)

        print('learning/indexing in progress ...')
        classifier.learn(data_source, None)
        print('learning/indexing completed')

        print('classification in progress ...')

        progress_counter = 0
        mistakes_count = 0
        current_total_attempts = 0

        local_attempts_count = 0
        local_mistakes_count = 0

        total_images_count = data_source.get_count()
        for image_index in xrange(0, total_images_count):

            # Dirty hack: element of data_source is excluded during classification to prevent comparing similar images.
            data_source.excluded_index = -1;
            image = data_source.get_image(image_index)
            actual_class = data_source.get_image_class(image_index)
            data_source.excluded_index = image_index
            calculated_class = classifier.classify_image(image)

            is_correct = calculated_class == actual_class
            if not is_correct:
                mistakes_count += 1
                local_mistakes_count += 1

            current_total_attempts += 1
            local_attempts_count += 1

            progress_counter += 1
            if progress_counter % 10 == 0:
                current_correct_results = current_total_attempts - mistakes_count
                current_accuracy = (float(current_correct_results) / current_total_attempts) * 100
                print repr(progress_counter) \
                      + ' already classified... (current accuracy = ' \
                      + repr(current_accuracy)\
                      + ') (increment: ' \
                      + repr(local_attempts_count-local_mistakes_count) \
                      + ' out of ' \
                      + repr(local_attempts_count) \
                      + ')'

                local_attempts_count = 0
                local_mistakes_count = 0

        data_source.excluded_index = -1

        print('classification completed')

        correct_results = current_total_attempts - mistakes_count
        accuracy = (float(correct_results) / current_total_attempts) * 100

        return accuracy

    @staticmethod
    def reset_environment():
        numpy.random.seed(12345)
        random.seed(12345)

    @staticmethod
    def __get_images_names(directory_path, extension):
        postfix = '.' + extension
        images_names = [image_name
                        for image_name
                        in os.listdir(directory_path)
                        if image_name.endswith(postfix)]
        return images_names
