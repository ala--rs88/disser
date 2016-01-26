import sys
import os
import time
import operator
import numpy
from skimage.feature import greycomatrix
from skimage.io import imread
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.cluster import KMeans
import math


__author__ = 'Igor'

IMAGES_DIR_PATH = 'brodatz_database_bd.gidx'
IMAGES_EXTENSION = 'png'


PQ_product_members_count = 5
PQ_clusters_count = 200


def get_images_names(directory_path, extension):
    postfix = '.' + extension
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

def readImage(image_path, color_depth):
    image = imread(image_path)
    bin_size = 256 / color_depth
    for i in xrange(213):
        for j in xrange(213):
            image[i, j] = image[i, j] / bin_size
    return image

# descriptors
# --------------------------------------------------------------------------
glcm_cache = {}
glcm_PCA_cache = {}
glcm_BINNED_PCA_cache = {}
glcm_PQ_Cache = {}
glcm_PQ_Codebooks = {}
glcm_PQ_Precomputed_Centroids_Squared_Distances = {}

def get_GLCM(image_name):
    descriptor = glcm_cache[image_name][:, :, 0, 0]
    return descriptor

def get_GLCM_PCA(image_name):
    descriptor = glcm_PCA_cache[image_name]
    return descriptor

def get_GLCM_PQ(image_name):
    descriptor = glcm_PQ_Cache[image_name]
    return descriptor

def get_descriptor(image_name):
    # DESCRIPTOR CONFIG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #descriptor = get_GLCM_PCA(image_name)
    descriptor = get_GLCM_PQ(image_name)
    # DESCRIPTOR CONFIG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return descriptor

# optimize: use matrix computation
# try using sklearn for metric computation
# try using l1 metric
def get_l2_distance(stored_descriptor, query_descriptor):
    return numpy.linalg.norm(stored_descriptor - query_descriptor)

# Both stored_descriptor and query_descriptor are PQ-descriptors
def get_PQ_distance(stored_descriptor, query_descriptor, product_members_count):

    distances_sum = 0
    for index in xrange(0, len(stored_descriptor)):
        stored_centroid_marker = stored_descriptor[index]
        query_centroid_marker = query_descriptor[index]

        a = stored_centroid_marker
        b = query_centroid_marker
        if a > b:
            a = query_centroid_marker
            b = stored_centroid_marker

        squared_distance = glcm_PQ_Precomputed_Centroids_Squared_Distances[index][a][b]
        distances_sum += squared_distance

    distance = math.sqrt(distances_sum)
    return distance

def calculate_distance(stored_descriptor, query_descriptor):
    # DISTANCE CONFIG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #return get_l2_distance(stored_descriptor, query_descriptor)
    return get_PQ_distance(stored_descriptor, query_descriptor, PQ_product_members_count)
    # DISTANCE CONFIG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



def getGLCM(images_dir_path, image_name):
    #image = readImage(os.path.join(images_dir_path, image_name), 16)
    image = imread(os.path.join(images_dir_path, image_name))
    glcm = greycomatrix(image, [5], [0], 256, symmetric=True, normed=False)
    return glcm


def precompute_GLCM_Cache(images_dir_path, images_names):
    print '--------------------------------------------------------------------------'
    start = time.time()
    for image_name in images_names:
        computedGLCM = getGLCM(images_dir_path, image_name)
        glcm_cache[image_name] = computedGLCM

    end = time.time()
    secs = end - start
    msecs = secs * 1000  # millisecs
    print 'GLCMs cache size:' + repr(sys.getsizeof(glcm_cache)) + ' bytes'
    print 'GLCMs cache dim:' + repr(len(glcm_cache.keys())) + '*' + repr(len(glcm_cache[glcm_cache.keys()[0]]))
    print 'GLCMs descriptors size:' + repr(sys.getsizeof(glcm_cache.values())) + ' bytes'
    print 'GLCM elapsed time: %f s (%f ms)' % (secs, msecs)
    print '--------------------------------------------------------------------------'

def precompute_GLCM_PCA_Cache(images_dir_path, images_names):
    print '--------------------------------------------------------------------------'
    start = time.time()
    flattened_descriptors = [None] * len(images_names);
    for i in xrange(len(images_names)):
        image_name = images_names[i]
        raw_descriptor = getGLCM(images_dir_path, image_name)
        flattened_descriptors[i] = raw_descriptor.flatten()

    PCA_train_set = numpy.array(flattened_descriptors)

    pca = PCA(n_components=0.8)
    print 'RAW:'
    print PCA_train_set.shape
    print PCA_train_set
    print ''
    transformedTrainSet = pca.fit_transform(PCA_train_set)
    print 'PCAed:'
    print transformedTrainSet.shape
    print transformedTrainSet
    print ''

    end = time.time()
    secs = end - start
    msecs = secs * 1000  # millisecs

    for i in xrange(len(images_names)):
        image_name = images_names[i]
        glcm_PCA_cache[image_name] = transformedTrainSet[i]

    print 'PCA GLCMs cache size:' + repr(sys.getsizeof(glcm_PCA_cache)) + ' bytes'
    print 'PCA GLCMs cache dim:' + repr(len(glcm_PCA_cache.keys())) + '*' + repr(len(glcm_PCA_cache[glcm_PCA_cache.keys()[0]]))
    print 'PCA GLCMs descriptors size:' + repr(sys.getsizeof(glcm_PCA_cache.values())) + ' bytes'
    print 'PCA GLCM elapsed time: %f s (%f ms)' % (secs, msecs)
    print '--------------------------------------------------------------------------'

def split_array_of_arrays_by_columns(array_of_arrays, chunks_count):
    rows_count, cols_count = array_of_arrays.shape

    columns_in_chunk_count = (cols_count + chunks_count - 1) / chunks_count

    chunks = [None] * chunks_count

    for chunk_index in xrange(0, chunks_count):
        first_col_index = chunk_index * columns_in_chunk_count;
        actual_chunk_size = min(columns_in_chunk_count, cols_count - first_col_index)
        chunks[chunk_index] = array_of_arrays[:,first_col_index:(first_col_index + actual_chunk_size)]

    return chunks


def compute_all_possible_squared_distances_combination(centroids_values):
    sq_distances = {}

    for a in xrange(0, len(centroids_values)):
        sq_distances[a] = {}
        for b in xrange(a, len(centroids_values)):
            c1 = centroids_values[a]
            c2 = centroids_values[b]
            distance = get_l2_distance(c1, c2)
            sq_distances[a][b] = distance**2

    return sq_distances;


def precompute_GLCM_Product_Quantization_Cache(images_dir_path, images_names, clusters_count, product_members_count):
    print '--------------------------------------------------------------------------'
    start = time.time()
    flattened_descriptors = [None] * len(images_names);
    for i in xrange(len(images_names)):
        image_name = images_names[i]
        raw_descriptor = getGLCM(images_dir_path, image_name)
        flattened_descriptors[i] = raw_descriptor.flatten()

    train_set = numpy.array(flattened_descriptors)

    chunks = split_array_of_arrays_by_columns(train_set, product_members_count)
    #glcm_PQ_Codebooks = [None]*len(chunks)
    print 'len(chunks)=' + repr(len(chunks))
    print 'product_members_count=' + repr(product_members_count)
    for image_name in images_names:
        glcm_PQ_Cache[image_name] = [None]*product_members_count

    for chunk_index in xrange(0, len(chunks)):
        chunk = chunks[chunk_index]
        estimator = KMeans(n_clusters=clusters_count)
        estimator.fit(chunk)

        glcm_PQ_Codebooks[chunk_index] = {
            'clusters_labels': estimator.labels_,
            'clusters_centriods': estimator.cluster_centers_,
            'estimator': estimator
        };
        glcm_PQ_Precomputed_Centroids_Squared_Distances[chunk_index] = compute_all_possible_squared_distances_combination(
            estimator.cluster_centers_)

        for image_index in xrange(0, len(estimator.labels_)):
            image_name = images_names[image_index]
            image_descriptor = glcm_PQ_Cache[image_name]
            image_descriptor[chunk_index] = estimator.labels_[image_index]

        print 'chunk calculated: ' + repr(chunk_index)

    end = time.time()
    secs = end - start
    msecs = secs * 1000  # millisecs

    print 'Product Quantization GLCMs cache size:' + repr(sys.getsizeof(glcm_PQ_Cache)) + ' bytes'
    print 'Product Quantization GLCMs cache dim:' + repr(len(glcm_PQ_Cache.keys())) + '*' + repr(len(glcm_PQ_Cache[glcm_PQ_Cache.keys()[0]]))
    print 'Product Quantization GLCMs descriptors size:' + repr(sys.getsizeof(glcm_PQ_Cache.values())) + ' bytes'
    print 'Product Quantization GLCMs distances size:' \
          + repr(sys.getsizeof(glcm_PQ_Precomputed_Centroids_Squared_Distances))\
          + ' bytes'
    print 'Product Quantization GLCM elapsed time: %f s (%f ms)' % (secs, msecs)
    print '--------------------------------------------------------------------------'

# -------------------------------------

def predict_PQ_descriptor(glcm_descriptor):
    flattened_glcm_descriptor = glcm_descriptor.flatten()

    query_descriptor_chunks = split_array_of_arrays_by_columns(numpy.array([flattened_glcm_descriptor]), PQ_product_members_count)

    pq_descriptor = [None]*len(query_descriptor_chunks)

    for index in xrange(0, len(query_descriptor_chunks)):
        print repr(index) + ' STARTED'
        glcm_chunk = query_descriptor_chunks[index]
        print 'chunk = ' + repr(glcm_chunk)
        estimator = glcm_PQ_Codebooks[index]['estimator']
        prediction = estimator.predict(glcm_chunk)
        centroid = prediction[0]
        print 'centroid = ' + repr(centroid)
        # !!! centroid means LABEL here, not actual centroid
        pq_descriptor[index] = centroid
        print repr(index) + ' OK'

    return pq_descriptor


def calculate_distances_descriptors_PQ(images_names, image_name_to_be_compared):
    descriptors = []

    query_GLCM_descriptor = get_GLCM(image_name_to_be_compared)
    query_descriptor = predict_PQ_descriptor(query_GLCM_descriptor)
    #query_descriptor = get_descriptor(image_name_to_be_compared)

    for image_name in images_names:
        if image_name == image_name_to_be_compared:
            continue
        stored_descriptor = get_descriptor(image_name)

        #start = time.time()
        distance = calculate_distance(stored_descriptor, query_descriptor)
        #end = time.time()
        #secs = end - start
        #msecs = secs * 1000  # millisecs
        #print 'PQ Distance calculation time: %f s (%f ms)' % (secs, msecs)

        descriptors.append((image_name_to_be_compared, image_name, distance))

    return descriptors


def calculate_distances_descriptors(images_names, image_name_to_be_compared):
    descriptors = []

    query_descriptor = get_descriptor(image_name_to_be_compared)

    for image_name in images_names:
        if image_name == image_name_to_be_compared:
            continue
        stored_descriptor = get_descriptor(image_name)
        distance = calculate_distance(stored_descriptor, query_descriptor)
        descriptors.append((image_name_to_be_compared, image_name, distance))

    return descriptors


def try_classify_image(images_names, image_name_to_be_classified):
    actual_image_class = get_class_by_image_name(image_name_to_be_classified)
    # print 'Image to be classified: (actual_class=' + actual_image_class + ') ' + image_name_to_be_classified

    #distances_descriptors = calculate_distances_descriptors(images_names, image_name_to_be_classified)
    distances_descriptors = calculate_distances_descriptors_PQ(images_names, image_name_to_be_classified)
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


    precompute_GLCM_Cache(IMAGES_DIR_PATH, images_names)
    #precompute_GLCM_PCA_Cache(IMAGES_DIR_PATH, images_names)

    precompute_GLCM_Product_Quantization_Cache(IMAGES_DIR_PATH, images_names, PQ_clusters_count, PQ_product_members_count)

    print ''

    total_classification_time_in_secs = 0
    classification_requests_count = 0

    progress_counter = 0
    mistakes_count = 0
    images_names_to_be_analyzed = images_names[:]
    for image_name_to_be_classified in images_names_to_be_analyzed:
        start = time.time()
        is_correct = try_classify_image(images_names, image_name_to_be_classified)
        end = time.time()
        secs = end - start

        total_classification_time_in_secs += secs
        classification_requests_count += 1

        if not is_correct:
            mistakes_count += 1

        progress_counter += 1
        if progress_counter % 10 == 0:
            print repr(progress_counter) + ' already classified...'


    total_attempts = len(images_names_to_be_analyzed)
    correct_results = total_attempts - mistakes_count
    accuracy = (float(correct_results) / total_attempts) * 100

    print 'Accuracy: ' + repr(accuracy) + '% (' + repr(correct_results) + ' out of ' + repr(total_attempts) + ')'
    print 'Total classification time: %f s (%f ms)' % (total_classification_time_in_secs, total_classification_time_in_secs * 1000)
    average_secs = total_classification_time_in_secs / classification_requests_count
    print 'Average classification time: %f s (%f ms)' % (average_secs, average_secs * 1000)

    # for distance_descriptor in distances_descriptors:
    #     print distance_descriptor[0] + ' to ' + distance_descriptor[1] + ': ' + repr(distance_descriptor[2])

if __name__ == '__main__':
    main()