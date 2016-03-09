from PIL import Image
from numpy import *
from skimage.feature import greycomatrix
from skimage.feature import local_binary_pattern
import os

__author__ = 'Igor'


# def pca(X):
#   # Principal Component Analysis
#   # input: X, matrix with training data as flattened arrays in rows
#   # return: projection matrix (with important dimensions first),
#   # variance and mean
#
#   #get dimensions
#   num_data,dim = X.shape
#
#   #center data
#   mean_X = X.mean(axis=0)
#   for i in range(num_data):
#       X[i] -= mean_X
#
#   if dim>100:
#       print 'PCA - compact trick used'
#       M = dot(X,X.T) #covariance matrix
#       e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
#       tmp = dot(X.T,EV).T #this is the compact trick
#       V = tmp[::-1] #reverse since last eigenvectors are the ones we want
#       S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
#   else:
#       print 'PCA - SVD used'
#       U,S,V = linalg.svd(X)
#       V = V[:num_data] #only makes sense to return the first num_data
#
#   #return the projection matrix, the variance and the mean
#   return V,S,mean_X
#



def main():

    filename = os.path.join("evaluation_results", "test_results")

    # try:
    #     os.remove(filename)
    # except OSError:
    #     pass

    with open(filename, 'a') as file:
        file.write("A")

    with open(filename, 'a') as file:
        file.write("B")

    # image = [[1, 2, 2, 2],
    #          [9, 5, 6, 4],
    #          [5, 3, 1, 16],
    #          [9, 7, 6, 4]]
    #
    # neighbours_count = 8
    # max_pixel_descriptor_value = 2**neighbours_count
    #
    # lbp_matrix = local_binary_pattern(image, neighbours_count, 1, 'ror')
    # print lbp_matrix
    #
    # hist, bin_edges = histogram(lbp_matrix, bins=range(max_pixel_descriptor_value + 1))
    # print hist



    # glcms = greycomatrix(image, [1], [0, (1./4)*pi, (1./2)*pi, (3./4)*pi], 3, symmetric=False, normed=False)
    #
    # print (glcms[:, :, 0, 0])
    # print (glcms[:, :, 0, 1]).flatten()
    # print (glcms[:, :, 0, 2]).flatten()
    # print (glcms[:, :, 0, 3]).flatten()
    # print (glcms[:, :, 0, 0] + glcms[:, :, 0, 1] + glcms[:, :, 0, 2] + glcms[:, :, 0, 3]).flatten()

if __name__ == '__main__':
    main()