__author__ = 'IgorKarpov'

import numpy as np
from sklearn.cluster import KMeans

def split_array_of_arrays_by_columns(array_of_arrays, chunks_count):
    rows_count, cols_count = array_of_arrays.shape

    columns_in_chunk_count = (cols_count + chunks_count - 1) / chunks_count

    chunks = [None] * chunks_count

    for chunk_index in xrange(0, chunks_count):
        first_col_index = chunk_index * columns_in_chunk_count;
        actual_chunk_size = min(columns_in_chunk_count, cols_count - first_col_index)
        chunks[chunk_index] = array_of_arrays[:,first_col_index:(first_col_index + actual_chunk_size)]

    return chunks

vectors = np.array([[1.,  2., 0.],
   [ 1.,  3., 0.],
   [ 1.,  4., 0.],
   [ 1.,  5., 0.],
   [ 7.,  6., 0.],
   [ 6.,  7., 0.],
   [ 7.,  8., 0.],
   [ 8.,  9., 0.],
   [ 9.,  10., 0.],
   [ 10.,  11., 0.]])

# x = split_array_of_arrays_by_columns(np.array([[1, 2, 3, 4, 5]]), 3);
# for xx in x:
#     print xx[0]
#     print ''

vectors = vectors[:,0:2]
print vectors

kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
# print 'y:'
# print y
print ''
print(kmeans.labels_)
print(kmeans.cluster_centers_)


x = kmeans.predict([1., 2.])
print 'x:'
print x

# import numpy as np
# import matplotlib.pyplot as plt
#
# N = 1
# ind = np.arange(N)  # the x locations for the groups
# width = 0.28       # the width of the bars
# fig, ax = plt.subplots()
#
# custom_oldest_means = (0.022537)
# rects1 = ax.bar(ind, custom_oldest_means, width, color='r')
#
# custom_oldest_random = (0.159854)
# rects2 = ax.bar(ind+width, custom_oldest_random, width, color='y')
#
# valgrind = (0.000156126)
# rects3 = ax.bar(ind+width+width, valgrind, width, color='b')
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Access time, ns')
# ax.set_title('Random reads access time')
# ax.set_xticks(ind+width)
# ax.set_xticklabels((''))
# ax.set_ylim(0,0.18)
#
# ax.legend((rects1[0], rects2[0], rects3[0]), ('SSD', 'HDD', 'RAM') ,loc='center right')
#
# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, "{:3.5f}".format(height),
#                 ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
#
# plt.show()