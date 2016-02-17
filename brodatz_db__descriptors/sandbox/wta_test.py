import itertools
import random
import math
import numpy

__author__ = 'IgorKarpov'

# array = list(range(8**2))
#
# permuts = itertools.permutations(array)


# for permut in permuts:
#    print permut
#    break
#
# f = math.factorial(256**2)
#
# print f
# print random.randint(0, f-1)
#
# # print (4**2)
# print (math.factorial(4**2))
#
# indexes = xrange(math.factorial(8**2))
# x = random.sample(indexes, 4)
# #x = indexes
# print('done')

# x = range(5)
# print x[:3]

N = 256**2
M = 10
K = 30

linIdx = range(N)
print 'linIdx = ' + repr(linIdx)

listWTAIndex = []
for i in xrange(M):
    listWTAIndex.append(numpy.random.permutation(linIdx)[:K])

# print 'listWTAIndex = ' + repr(listWTAIndex)
print 'OK'