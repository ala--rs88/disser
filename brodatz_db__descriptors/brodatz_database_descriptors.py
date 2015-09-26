__author__ = 'Igor'

import os

IMAGES_DIR_PATH = 'brodatz_database_bd.gidx'
IMAGES_EXTENSION = 'png'


def get_images_names(directory_path, extension):
    postfix = '.' + extension;
    images_names = [image_name for image_name in os.listdir(directory_path) if image_name.endswith(postfix)]
    return images_names


def main():
    images_names = get_images_names(IMAGES_DIR_PATH, IMAGES_EXTENSION)
    print len(images_names)

if __name__ == '__main__':
    main()