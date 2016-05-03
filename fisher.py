import numpy as np
import matplotlib.pyplot as plt
from faces import FaceSpace
from imagedatabase import ImageDatabase

class FisherSpace(FaceSpace):
	"""
	A face space for using the Fisherfaces method.
	"""
	def __init__(self, images, dimensions=(360, 256)):
		super(FisherSpace, self).__init__(images, dimensions)

		self.people = np.column_stack(self.image_vectors)
		self.class_averages = np.cumsum(self.people, axis=1)[:,7::8] / 8
		self.class_averages[1:] = self.class_averages[1:] - self.class_averages[:-1]
		
		self.normal_people = self.people - np.repeat(self.class_averages, 8, axis=1)
		self.within_class = self.normal_people @ self.normal_people.T

		between_matrix = self.class_averages - np.repeat(self.flatten(self.average_face), 43, axis=1)
		self.between_class = 8 * between_matrix @ between_matrix.T

		S = np.linalg.inv(self.within_class) @ self.between_class
		self.eigenvals, self.eigenvects = np.linalg.eig(S)

imdb = ImageDatabase(directory="database", resample=(90, 64))

fisherfaces = FisherSpace(imdb)