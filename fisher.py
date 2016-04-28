import numpy as np
import matplotlib.pyplot as plt
from faces import FaceSpace
from imagedatabase import ImageDatabase

class FisherSpace(FaceSpace):

	def __init__(self, images, dimensions=(360, 256)):
		super(FisherSpace, self).__init__(images, dimensions)

		slices = [(x * 8, (x + 1) * 8) for x in range(43)]
		self.people = [self.image_vectors[slice(*x)] for x in slices]
		self.class_averages = [self._average_face(person) for person in self.people]
		self.normal_people = [image - self.class_averages[i // 8] for i, image in enumerate(self.image_vectors)]

		stacked = np.column_stack(self.normal_people)
		self.between_class = np.dot(stacked, stacked.T)


imdb = ImageDatabase(directory="database", resample=(90, 64))

fisherfaces = FisherSpace(imdb)