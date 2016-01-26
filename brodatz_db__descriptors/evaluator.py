from data_source import DataSource
#from finders.glcm_finder import GLCMFinder
from finders.glcm_pca_finder import GLCMPCAFinder
from finders.glcm_pq_symm_finder import GLCMPQSymmetricFinder
from knn_classifier import kNNClassifier
import os

__author__ = 'IgorKarpov'


class Evaluator:

    files_path = None

    def __init__(self, files_path):
        self.files_path = files_path

    def evaluate_accuracy(self):
        file_names = self.__get_images_names(self.files_path, 'png')
        data_source = DataSource(self.files_path, file_names)
        #finder = GLCMFinder(data_source)
        #finder = GLCMPCAFinder(data_source)
        finder = GLCMPQSymmetricFinder(data_source, 5, 200)
        classifier = kNNClassifier(5, finder)

        print('learning/indexing in progress ...')
        classifier.learn(data_source, None)
        print('learning/indexing completed')

        print('classification in progress ...')

        progress_counter = 0
        mistakes_count = 0
        current_total_attempts = 0

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

            current_total_attempts += 1

            progress_counter += 1
            if progress_counter % 10 == 0:
                current_correct_results = current_total_attempts - mistakes_count
                current_accuracy = (float(current_correct_results) / current_total_attempts) * 100
                print repr(progress_counter) + ' already classified... (current accuracy = ' + repr(current_accuracy) + ')'

        data_source.excluded_index = -1;

        print('classification completed')

        correct_results = current_total_attempts - mistakes_count
        accuracy = (float(correct_results) / current_total_attempts) * 100

        return accuracy

    def __get_images_names(self, directory_path, extension):
        postfix = '.' + extension
        images_names = [image_name
                        for image_name
                        in os.listdir(directory_path)
                        if image_name.endswith(postfix)]
        return images_names