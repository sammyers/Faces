import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import os
import math

class EigenFaces(object):
    """

    """
    def __init__(self, image_database):
        super(EigenFaces, self).__init__()
        self.imdb = image_database

        self.image_dimensions = (360, 256)

    def flatten(self, image):
        """
        Take a m x n image matrix and return a (m*n) x 1 image vector.
        """
        return np.reshape(image, (image.size))

    def average_face(self, image_list):
        """
        Take a list of image vectors and return the 'average' of those.
        """
        matrix = np.column_stack(image_list)
        return matrix.sum(axis=1) / matrix.shape[1]

    def normalize_faces(self, image_list):
        """
        Take a list of image vectors and return a matrix with normalized columns.
        """
        avg = self.average_face(image_list)
        images = [image - avg for image in image_list]
        images = np.matrix(np.column_stack(images))
        return images

    def get_covariance_matrix(self, image_list, order='small'):
        matrix = self.normalize_faces(image_list)
        M = len(image_list)

        if order == 'large':
            return matrix * matrix.T / M
        elif order == 'small':
            return matrix.T * matrix / M
        else:
            raise Exception("Undefined order: '{}'".format(order))

    def get_eigenfaces(self, image_list):
        """
        Return a list of column vectors representing the eigenfaces of a set of images.
        """
        eigenthings = np.linalg.eig(self.get_covariance_matrix(image_list, order='small'))
        vectors = self.normalize_faces(image_list) * eigenthings[1]

        return np.hsplit(vectors, vectors.shape[1])

    def show_eigenfaces(self, limit=16):
        """
        Display some number of eigenfaces corresponding to the largest eigenvalues
        for a training set in the specified directory.
        """
        eigenfaces = self.get_eigenfaces([self.flatten(image) for image in images])
        reshaped_eigenfaces = [np.reshape(face, self.image_dimensions) for face in eigenfaces]

        self.show_grid(reshaped_eigenfaces[:limit - 1])
    
    def show_grid(self, images):
        rows = int(math.floor(len(images) ** 0.5))
        cols = len(images) / rows


        fig, axes = plt.subplots(rows, cols)
        for i, image in images:
            a, b = i / cols, i % cols
            axes[a, b].imshow(image, cmap=plt.cm.gray)
            axes[a, b].set_axis_off()
        plt.show()


if __name__ == '__main__':
    show_eigenfaces('../database')