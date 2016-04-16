import numpy as np
import matplotlib.pyplot as plt
import sympy

def center_data(data):
    """
    Take a column of data and return a column of the same size with the mean subtracted.
    """
    return (data - np.mean(data)) / np.sqrt(len(data) - 1)

def covariance(*columns):
    """
    Take any number of 1D data arrays and return the covariance matrix.
    """
    columns = [center_data(column) for column in columns]
    data_matrix = np.matrix(np.column_stack(columns))
    return data_matrix.T * data_matrix


def flatten(image):
    """
    Take a m x n image vector and return a (m*n) x 1 image vector.
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
    return images

def get_covariance_matrix(image_list, order='small'):
    normalized = normalize_faces(image_list)
    matrix = np.matrix(np.column_stack(image_list))
    if order == 'large':
        return matrix * matrix.T / len(image_list)
    return matrix.T * matrix / len(image_list)
