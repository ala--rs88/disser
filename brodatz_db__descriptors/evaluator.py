from data_source import DataSource
from finders.glcm_finder import GLCMFinder
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
        finder = GLCMFinder(data_source)
        classifier = kNNClassifier(5, finder)
        classifier.learn(data_source, None)

        progress_counter = 0
        mistakes_count = 0
        total_attempts = 0

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

            progress_counter += 1
            if progress_counter % 10 == 0:
                print repr(progress_counter) + ' already classified...'

            total_attempts += 1

        data_source.excluded_index = -1;


        correct_results = total_attempts - mistakes_count
        accuracy = (float(correct_results) / total_attempts) * 100

        return accuracy

    def __get_images_names(self, directory_path, extension):
        postfix = '.' + extension
        images_names = [image_name
                        for image_name
                        in os.listdir(directory_path)
                        if image_name.endswith(postfix)]
        return images_names