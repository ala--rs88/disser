__author__ = 'IgorKarpov'

import os
from skimage.io import imread


class DataSource:

    images_files_path = None
    images_files_names = []

    excluded_index = -1

    def __init__(self, images_files_path, images_files_names):
        self.images_files_path = images_files_path
        self.images_files_names = images_files_names

    def get_count(self):
        visible_count = len(self.images_files_names)
        if self.excluded_index >= 0:
            visible_count -= 1
        return visible_count

    def get_image_file_name(self, image_index):
        actual_image_index = self.__convert_visible_index_to_actual(image_index)
        image_file_name = self.images_files_names[actual_image_index]
        return image_file_name

    def get_image(self, image_index):
        actual_image_index = self.__convert_visible_index_to_actual(image_index)
        image_file_name = self.images_files_names[actual_image_index]
        path = os.path.join(self.images_files_path, image_file_name)
        image = self.__readAndBinImage(path)
        #image = imread(path)
        return image

    def get_image_class(self, image_index):
        actual_image_index = self.__convert_visible_index_to_actual(image_index)
        image_file_name = self.images_files_names[actual_image_index]
        image_class = image_file_name[1:4]
        return image_class

    def __convert_visible_index_to_actual(self, visible_index):
        actual = visible_index
        if 0 <= self.excluded_index <= visible_index:
            actual = visible_index + 1
        return actual

    @staticmethod
    def __readAndBinImage(image_path):
        image = imread(image_path)
        x, y = image.shape
        color_depth = 8
        bin_size = 256 / color_depth
        for i in xrange(x):
            for j in xrange(y):
                image[i, j] = image[i, j] / bin_size
        return image