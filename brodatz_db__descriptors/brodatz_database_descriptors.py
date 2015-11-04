import sys
import os
import operator
import numpy
from skimage.feature import greycomatrix
from skimage.io import imread

__author__ = 'Igor'

IMAGES_DIR_PATH = 'brodatz_database_bd.gidx'
IMAGES_EXTENSION = 'png'


def get_images_names(directory_path, extension):
    postfix = '.' + extension;
    images_names = [image_name for image_name in os.listdir(directory_path) if image_name.endswith(postfix)]
    return images_names


def get_class_by_image_name(image_name):
    image_class = image_name[1:4]
    return image_class


def get_prevailing_class(image_names):
    occurrences = {}
    for image_name in image_names:
        image_class = get_class_by_image_name(image_name)
        if image_class in occurrences:
            occurrences[image_class] += 1
        else:
            occurrences[image_class] = 1

    prevailing_class = max(occurrences.iteritems(), key=operator.itemgetter(1))[0]
    return prevailing_class

# cache
# --------------------------------------------------------------------------
glcm_cache = {}
def getGLCM( image_name):
    # if not image_name in glcm_cache:
    #     image = imread(os.path.join(images_dir_path, image_name))
    #     computedGLCM = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False)
    #     glcm_cache[image_name] = computedGLCM

    glcm = glcm_cache[image_name]
    return glcm

def precomputeGLCMCache(images_dir_path, images_names):
    for image_name in images_names:
        image = imread(os.path.join(images_dir_path, image_name))
        computedGLCM = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False)
        glcm_cache[image_name] = computedGLCM

# -------------------------------------

def calculate_distances_descriptors(images_names, image_name_to_be_compared):
    descriptors = []

    glcm_to_be_compared = getGLCM(image_name_to_be_compared)

    for image_name in images_names:
        if image_name == image_name_to_be_compared:
            continue
        glcm = getGLCM(image_name)
        distance = numpy.linalg.norm(glcm[:, :, 0, 0] - glcm_to_be_compared[:, :, 0, 0])
        descriptors.append((image_name_to_be_compared, image_name, distance))

    return descriptors


def try_classify_image(images_names, image_name_to_be_classified):
    actual_image_class = get_class_by_image_name(image_name_to_be_classified)
    # print 'Image to be classified: (actual_class=' + actual_image_class + ') ' + image_name_to_be_classified

    distances_descriptors = calculate_distances_descriptors(images_names, image_name_to_be_classified)
    distances_descriptors.sort(key=lambda tup: tup[2])

    top_closest_images_names = [d[1] for d in distances_descriptors[:5]]
    computed_class = get_prevailing_class(top_closest_images_names)

    is_classification_correct = actual_image_class == computed_class

    # if is_classification_correct:
    #     print 'CORRECT',
    # else:
    #     print 'MISTAKE',
    # print 'Computed class: ' + computed_class + '(top_similar: ',
    # for top_image_name in top_closest_images_names:
    #     print top_image_name,
    # print ')'

    return is_classification_correct


def main():
    images_names = get_images_names(IMAGES_DIR_PATH, IMAGES_EXTENSION)
    print 'Images count: ' + repr(len(images_names))


    precomputeGLCMCache(IMAGES_DIR_PATH, images_names)
    print 'GLCMs cache size:' + repr(sys.getsizeof(glcm_cache) / 1024) + ' kilobytes'
    print ''

    progress_counter = 0
    mistakes_count = 0
    images_names_to_be_analyzed = images_names[:]
    for image_name_to_be_classified in images_names_to_be_analyzed:
        is_correct = try_classify_image(images_names, image_name_to_be_classified)
        if not is_correct:
            mistakes_count += 1

        progress_counter += 1
        if progress_counter % 10 == 0:
            print repr(progress_counter) + ' already classified...'


    total_attempts = len(images_names_to_be_analyzed)
    correct_results = total_attempts - mistakes_count
    accuracy = (float(correct_results) / total_attempts) * 100

    print 'Accuracy: ' + repr(accuracy) + '% (' + repr(correct_results) + ' out of ' + repr(total_attempts) + ')'

    # for distance_descriptor in distances_descriptors:
    #     print distance_descriptor[0] + ' to ' + distance_descriptor[1] + ': ' + repr(distance_descriptor[2])

if __name__ == '__main__':
    main()
