#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.distance import euclidean

class EigenSpace(object):
    """
    An object representing the 'face space' for a set of images.
    """
    def __init__(self, images, dimensions=(360, 256)):
        super(EigenSpace, self).__init__()

        self.image_dimensions = dimensions

        self.new_dimensions = (dimensions[0] / 4, dimensions[1] / 4)

        self.images = list(images)

        self.image_vectors = [self.flatten(image) for image in self.images]

        self.average_face = self._average_face()

        self.eigenvectors = self.get_eigenvectors()

    def flatten(self, image):
        """
        Take a m x n image matrix and return a (m*n) x 1 image vector.
        """
        return np.reshape(image, (image.size, 1))

    def _average_face(self):
        """
        Take a list of image vectors and return the 'average' of those.
        """
            
        matrix = np.column_stack(self.image_vectors)
        return matrix.sum(axis=1) / matrix.shape[1]

    def normalize_faces(self, image_list):
        """
        Take a list of image vectors and return a matrix with normalized columns.
        """
        avg = self.average_face
        images = np.matrix(np.column_stack(image_list)) - np.column_stack([avg] * len(image_list))
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

    def get_eigenvectors(self, limit=None):
        """
        Return a list of column vectors representing the eigenfaces of a set of images.
        """
        eigenthings = np.linalg.eig(self.get_covariance_matrix(self.image_vectors))
        vectors = self.normalize_faces(self.image_vectors) * eigenthings[1]
        if limit == None:
            return vectors
        else:
            return vectors[:,:limit]

    def show_eigenfaces(self, limit=16):
        """
        Display some number of eigenfaces corresponding to the largest eigenvalues
        for a training set in the specified directory.
        """
        vectors = np.hsplit(self.eigenvectors, self.eigenvectors.shape[1])
        eigenfaces = [np.reshape(face, self.image_dimensions) for face in vectors]
        self.show_grid(eigenfaces[:limit - 1])
    
    def show_grid(self, images):
        rows = int(math.floor(len(images) ** 0.5))
        cols = len(images) / rows

        fig, axes = plt.subplots(rows, cols)

        for i, image in enumerate(images):
            a, b = i / cols, i % cols
            axes[a, b].imshow(image, cmap=plt.cm.gray)
            axes[a, b].set_axis_off()
        plt.show()

    def get_weight_vector(self, image):
        # import pdb; pdb.set_trace()

        normalized = self.flatten(image).T[0] - self.average_face
        projection = np.dot(self.eigenvectors.T, normalized)#np.vstack([np.dot(vector, normalized) for vector in self.eigenvectors])
        return projection

    def recognize_face(self, image):
        """
        Take an image and run facial recognition on it.
        If recognized as a face within the training set, return that face.
        """
        projection = self.get_weight_vector(image)
        # This assumes there is a match, update this later to include thresholds

        euclidean_distances = [euclidean(self.get_weight_vector(x), projection) for x in self.images]

        recognized_index, euclidean_distance = min(enumerate(euclidean_distances), key=lambda x: x[1])

        recognized_image = self.images[recognized_index]

        return (recognized_index, recognized_image, euclidean_distance)


def recognize_face_example(imdb):
    all_faces = EigenSpace(imdb.subset(remove='01'))

    # for emotion in imdb.iterate_emotions():
    #     e_faces.show_eigenfaces(emotion)
    # all_faces.show_eigenfaces()
    danny = imdb['faceimage_dannyWolf_01.png']

    return all_faces.recognize_face(danny)

def make_emotion_eigenspace(imdb):
    eigenemotions = [EigenSpace(emotion) for emotion in imdb.emotions()]

    return EigenSpace([np.reshape(e_emotion.get_eigenvectors(limit=1), (360, 256)) for e_emotion in eigenemotions])


if __name__ == '__main__':
    from imagedatabase import ImageDatabase

    imdb = ImageDatabase(directory="database")

    final_faces = ImageDatabase(directory="finalFaces")


    e_space = EigenSpace(imdb)

    print 'EigenSpace created'

    # face_matches = []

    # for i, f in enumerate(final_faces):
    #     print 'recognizing {}'.format(i)
    #     index, image, e_dist = e_space.recognize_face(f)
    #     face_matches.append(np.hstack((image, f)))
    #     plt.imshow(image, cmap=plt.cm.gray)
    #     plt.show()
    #     break

    # import pdb; pdb.set_trace()

    # e_space.show_grid(face_matches)


    index, image, e_dist = e_space.recognize_face(final_faces['faceimage_nathanielYee_00.png'])

    result = recognize_face_example(imdb)

    e_space.show_eigenfaces()

    print index

    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
