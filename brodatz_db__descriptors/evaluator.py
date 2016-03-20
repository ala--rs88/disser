from data_source import DataSource
from knn_classifier import kNNClassifier
import os
import numpy
import random
import sys
from multiprocessing import Process
import traceback
from uuid import uuid4
import time

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

def evaluate_data_for_result_file(finder_pack, descriptor_pack, image_depths_to_be_evaluated, files_path, file_names):
    try:
        finder_name = finder_pack['finder_name']
        finder_parameters_names = finder_pack['finder_parameters_names']
        finder_parameters_sets = finder_pack['finder_parameters_sets']
        build_finder = finder_pack['build_finder']
        descriptor_name = descriptor_pack['descriptor_name']
        descriptor_additional_parameters_names = descriptor_pack['descriptor_additional_parameters_names']
        descriptor_additional_parameters_sets = descriptor_pack['descriptor_additional_parameters_sets']
        build_descriptor_builder = descriptor_pack['build_descriptor_builder']

        full_result_file_path = Evaluator.get_full_path_for_file(finder_name, descriptor_name)
        if (os.path.exists(full_result_file_path)):
            print('FILE ALREADY EXISTS: ' + full_result_file_path + '\n')
            return

        Evaluator.prepare_result_file(full_result_file_path)

        columns_names = ['image_depth']
        columns_names.extend(finder_parameters_names)
        columns_names.extend(descriptor_additional_parameters_names)
        columns_names.extend(['accuracy',
                              'learning_time_s',
                              '1_image_classification_time_ms',
                              'total_classification_time_s'])
        columns_names_csv_row = ','.join(columns_names)
        with open(full_result_file_path, 'a') as result_file:
            result_file.write(columns_names_csv_row)
            result_file.write('\n')

        for image_depth in image_depths_to_be_evaluated:
            for finder_parameters_set in finder_parameters_sets:
                for descriptor_additional_parameters_set in descriptor_additional_parameters_sets:

                    csv_row_values = [repr(image_depth)]
                    finder_values = map(lambda x: repr(x), finder_parameters_set)
                    csv_row_values.extend(finder_values)
                    descriptor_values = map(lambda x: repr(x), descriptor_additional_parameters_set)
                    csv_row_values.extend(descriptor_values)

                    try:
                        Evaluator.reset_environment()
                        accuracy, learning_time_secs, average_classification_time_ms, total_classification_time_secs\
                            = Evaluator.evaluate_set_accuracy(
                                files_path,
                                file_names,
                                image_depth,
                                descriptor_additional_parameters_set,
                                build_descriptor_builder,
                                finder_parameters_set,
                                build_finder)
                        csv_row_values.extend([repr(accuracy),
                                               repr(learning_time_secs),
                                               repr(average_classification_time_ms),
                                               repr(total_classification_time_secs)])
                    except:
                        e = sys.exc_info()
                        print("Error type 3: " + repr(e) + '\n\n')
                        csv_row_values.extend(['error', 'error', 'error', 'error'])
                        additional_info = ','.join(csv_row_values)
                        log_error("TYPE_3", e, finder_name, descriptor_name, additional_info)


                    csv_row = ','.join(csv_row_values)
                    with open(full_result_file_path, 'a') as result_file:
                        result_file.write(csv_row)
                        result_file.write('\n')

                    print('SET EVALUATED: ' + full_result_file_path + '\n')

        print('FILE READY: ' + full_result_file_path + '\n')

    except:
        e = sys.exc_info()
        print("Error type 2: " + repr(e) + '\n\n')
        log_error("TYPE_2", e, 'NONE', 'NONE', None)


def log_error(error_label, error_info, finder_name, descriptor_name, additional_info):
    error_file_path = Evaluator.get_full_path_for_error_file(error_label, finder_name, descriptor_name)
    with open(error_file_path, 'w') as error_file:
        if not additional_info is None:
            error_file.write(additional_info + '\n\n')
        error_file.write(error_label + repr(error_info) + '\n\n')
        traceback.print_exception(*error_info, file=error_file)

class Evaluator:

    __files_path = None
    __file_names = None

    def __init__(self, files_path):
        self.__files_path = files_path
        self.__file_names = Evaluator.__get_images_names(files_path, 'png')
        # random.seed(12345)
        # self.__file_names = random.sample(self.__file_names, 50)
        #print self.__file_names
        print ''

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
    def evaluate_accuracy(self,
                          image_depths_to_be_evaluated,
                          descriptors_packs,
                          finders_packs):
        try:

            processes = []
            for finder_pack in finders_packs:
                for descriptor_pack in descriptors_packs:
                    process = Process(
                        target=evaluate_data_for_result_file,
                        args=[finder_pack, descriptor_pack, image_depths_to_be_evaluated, self.__files_path, self.__file_names])
                    processes.append(process)

            for process in processes:
                process.start()
            for process in processes:
                process.join()

            print('\n\nEVALUATION FINISHED\n\n')

        except:
            e = sys.exc_info()
            print("Error type 1: " + repr(e) + '\n\n')
            log_error("TYPE_1", e, 'NONE', 'NONE', None)

    @staticmethod
    def get_full_path_for_file(finder_name, descriptor_name):
        file_name = finder_name + '_' + descriptor_name
        full_path = os.path.join("evaluation_results", file_name)
        return full_path

    @staticmethod
    def get_full_path_for_error_file(error_label, finder_name, descriptor_name):
        uuid = uuid4()
        uuid_string = str(uuid)
        file_name = finder_name + '_' + descriptor_name + '_ERROR_' + error_label + '_' + uuid_string
        full_path = os.path.join("evaluation_results", file_name)
        return full_path


    @staticmethod
    def prepare_result_file(full_path_to_file):
        try:
            os.remove(full_path_to_file)
        except OSError:
            pass

    @staticmethod
    def evaluate_set_accuracy(files_path,
                              file_names,
                              image_depth,
                              descriptor_additional_parameters,
                              build_descriptor_builder,
                              finder_parameters_set,
                              build_finder):
        data_source = DataSource(files_path, file_names, image_depth)
        descriptor_builder = build_descriptor_builder(image_depth, descriptor_additional_parameters)
        finder = build_finder(data_source, descriptor_builder, finder_parameters_set)
        classifier = kNNClassifier(5, finder)

        print('learning/indexing in progress ...')
        learning_start = time.time()
        classifier.learn(data_source, None)
        learning_end = time.time()
        learning_time_secs = learning_end - learning_start
        print('learning/indexing completed')

        print('classification in progress ...')

        progress_counter = 0
        mistakes_count = 0
        current_total_attempts = 0

        local_attempts_count = 0
        local_mistakes_count = 0

        total_classification_time_secs = 0

        total_images_count = data_source.get_count()
        for image_index in xrange(0, total_images_count):

            # Dirty hack: element of data_source is excluded during classification to prevent comparing similar images.
            data_source.excluded_index = -1;
            image = data_source.get_image(image_index)
            actual_class = data_source.get_image_class(image_index)
            data_source.excluded_index = image_index

            single_classification_start = time.time()
            calculated_class = classifier.classify_image(image)
            single_classification_end = time.time()
            single_classification_time_secs = single_classification_end - single_classification_start
            total_classification_time_secs += single_classification_time_secs

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
        average_classification_time_ms = (total_classification_time_secs / current_total_attempts) * 1000

        return accuracy, learning_time_secs, average_classification_time_ms, total_classification_time_secs

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
