import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import os

def import_images(directory):
    filenames = os.listdir(directory)
    images = [imread('{}/{}'.format(directory, filename), flatten=True) for filename in filenames]
    return images

def flatten(image):
    """
    Take a m x n image matrix and return a (m*n) x 1 image vector.
    """
    return np.reshape(image, (image.size))

def average_face(image_list):
    """
    Take a list of image vectors and return the 'average' of those.
    """
    matrix = np.column_stack(image_list)
    return matrix.sum(axis=1) / matrix.shape[1]

def normalize_faces(image_list):
    """
    Take a list of image vectors and return a matrix with normalized columns.
    """
    avg = average_face(image_list)
    images = [image - avg for image in image_list]
    images = np.matrix(np.column_stack(images))
    return images

def get_covariance_matrix(image_list, order='small'):
    matrix = normalize_faces(image_list)
    M = len(image_list)
    if order == 'large':
        return matrix * matrix.T / M
    return matrix.T * matrix / M

def get_eigenfaces(image_list):
    """
    Return a list of column vectors representing the eigenfaces of a set of images.
    """
    eigenthings = np.linalg.eig(get_covariance_matrix(image_list, order='small'))
    vectors = normalize_faces(image_list) * eigenthings[1]
    return np.hsplit(vectors, vectors.shape[1])

def show_eigenfaces(directory, limit=15):
    """
    Display some number of eigenfaces corresponding to the largest eigenvalues
    for a training set in the specified directory.
    """
    row, col = 3, 5
    images = import_images(directory)
    eigenfaces = get_eigenfaces([flatten(image) for image in images])
    eigenfaces = [np.reshape(face, (360, 256)) for face in eigenfaces]
    fig, axes = plt.subplots(row, col)
    for i, image in enumerate(eigenfaces[:limit]):
        a, b = i / col, i % col
        axes[a, b].imshow(image, cmap=plt.cm.gray)
        axes[a, b].set_axis_off()
    plt.show()

if __name__ == '__main__':
    show_eigenfaces('../database')