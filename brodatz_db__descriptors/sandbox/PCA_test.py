__author__ = 'IgorKarpov'

import sys
import numpy as np
from sklearn.decomposition import PCA


dict = {}
array = []

# print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
# print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'
#
# for i in xrange(10):
#     dict[i] = i;
#     array.append(i)
# print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
# print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'
#
# for i in xrange(90):
#     dict[i] = i*100;
#     array.append(i*100)
# print 'dict size:' + repr(sys.getsizeof(dict)) + ' bytes'
# print 'array size:' + repr(sys.getsizeof(array)) + ' bytes'

#[ 4160  6240 58937 19271]
# [ 4160  6240 58937 19271]


m1 = np.array([[1, 2], [1, 2]])
m2 = np.array([[3, 2], [1, 2]])
m3 = np.array([[1, 2], [10, 2]])
m4 = np.array([[1, 2], [1, 2]])

q = np.array([[1, 2], [1, 2]])

X = np.array([
    m1.flatten(),
    m2.flatten(),
    m3.flatten(),
    m4.flatten()])

Q = np.array([q.flatten()])

print 'X:'
print X
print 'Q:'
print Q
print ''

pca = PCA(n_components=0.99)
pca.fit(X)

Y = pca.fit_transform(X)
print 'transformed X:'
print Y
print ''

t = pca.transform(q.flatten())
print 'transformed Q:'
print t
print ''



PCA(copy=True, n_components=2, whiten=False)

print(pca.explained_variance_ratio_)