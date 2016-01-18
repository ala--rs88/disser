__author__ = 'IgorKarpov'

import sys
import numpy as np
from sklearn.decomposition import PCA


dict = {}
array = []

print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'

for i in xrange(10):
    dict[i] = i;
    array.append(i)
print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'

for i in xrange(90):
    dict[i] = i*100;
    array.append(i*100)
print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'

#[ 4160  6240 58937 19271]
# [ 4160  6240 58937 19271]


# m1 = np.array([[4160, 6240], [58937, 19271]])
# m2 = np.array([[4160, 6240], [58937, 19271]])
# m3 = np.array([[4160, 6240], [58937, 19271]])
# m4 = np.array([[4160, 6240], [58937, 19271]])
#
# X = np.array([
#     m1.flatten(),
#     m2.flatten(),
#     m3.flatten(),
#     m4.flatten()])
#
# print X
# pca = PCA(n_components=4)
# pca.fit(X)
#
# Y = pca.fit_transform(X)
# print ''
# print Y
# print ''
# PCA(copy=True, n_components=2, whiten=False)

#print(pca.explained_variance_ratio_)