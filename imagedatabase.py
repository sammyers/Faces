#!/usr/bin/env python2

import os
import glob

import numpy as np
import scipy.misc

class ImageDatabase(object):
    """
    ImageDatabase

    Provides helper functionality for dealing with a folder of images.

    Assumptions about the images in the directory:

    - There are an equal number of images of each person
    - The names of the images are of the format "x_name_magnitude.type"

    """


    def __init__(self, directory='face_images'):
        super(ImageDatabase, self).__init__()

        # Defines the image filetype extension to load
        self.image_filetype = "png"

        # Convert the image to grayscale on load?
        self.use_grayscale = True

        # Cache images in a dictionary on load or reload every time?
        self.image_caching = True

        # Image file basenames have information split by a character
        self.split_char = "_"

        # The index of the person's name in the file basename
        self.name_idx = 1

        # The index of the 'emotion' in the file basename
        # this number only makes sense when sorted for each person, as the
        # ordering of images is a mess
        self.magnitude_idx = 2

        # Error out if the directory doesn't exist or contain images
        if not self._check_valid_directory(directory):
            raise Exception(
                "Invalid directory: non-existent or no images of type {}".format(self.image_filetype)
            )

        # Where are the images located?
        self.directory = directory

        # Sorted list of images in the directory at instantiation time
        self.image_list = sorted(self._get_image_list(self.directory))

        # If caching is enabled, here's where the images go, file basenames are the keys
        self.cached_images = {}


    def _get_image_list(self, directory):
        """
        Get all images of the right type in directory.

        The type is defined by self.image_filetype.
        """
        glob_path = os.path.join(directory, "*.{}".format(self.image_filetype))
        return [os.path.basename(f_name) for f_name in glob.glob(glob_path)]


    def _are_images_in_directory(self, directory):
        """
        Check to see if there are more than 0 images of the right type in given directory.
        """
        return len(self._get_image_list(directory)) > 0


    def _check_valid_directory(self, directory):
        return (os.path.isdir(directory) and
            self._are_images_in_directory(directory))


    def _load_image(self, name):
        """
        Loads an image using scipy into a numpy array,
        converts image into grayscale if setting enabled.
        """
        return scipy.misc.imread(
            os.path.join(self.directory, name),
            self.use_grayscale
        )


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, key):
        """
        Allows ImageDatabase to be accessed using the imgDB[filename] pattern.

        This is the defacto way of loading images, because images can be cached
        if need be.
        """
        if self.image_caching:
            # load from cache or cache and return
            if hasattr(self.cached_images, key):
                return self.cached_images[key]
            else:
                image = self._load_image(key)
                self.cached_images[key] = image
                return image
        else:
            return self._load_image(key)


    def __iter__(self):
        """
        Default iteration method. Yields each image individually.
        """
        for img in self.image_list:
            yield self[img]


    def _get_split_filenames(self):
        """
        Helper method for splitting each file basename by the split character.

        Allows for reasoning by different parts of the filename in other methods.
        """
        return [f_name.split(self.split_char) for f_name in self.image_list]


    def _get_people_set(self):
        """
        Helper method for iterating by person.

        Returns the set of names included in the images.
        """
        split_f_names = self._get_split_filenames()

        return set([split_f_name[self.name_idx] for split_f_name in split_f_names])


    def _iterate_people_helper(self):
        """
        Yields the list of split filenames for each image for each person in 
        the image database.

        For use in the iterate_people method and iterate_emotions method.
        """
        split_f_names = self._get_split_filenames()

        people = self._get_people_set()

        for person in people:
            images_of_person = filter(
                lambda split_f_name: split_f_name[self.name_idx] == person,
                split_f_names
            )

            yield images_of_person

    def iterate_people(self):
        """
        Iterate the images by person. Yields an array of images for each person.
        """
        for images_of_person in self._iterate_people_helper():

            # rejoin split string and yield array of images of the person
            yield [self[self.split_char.join(split_f_name)] for split_f_name in images_of_person]


    def iterate_emotions(self):
        """
        Iterate the images by emotion. Yields an array of images for each emotion.
        """
        def make_matrix():
            for split_f_name_list in self._iterate_people_helper():
                yield [self.split_char.join(split_f_name) for split_f_name in split_f_name_list]

        # [names x emotions]
        filename_matrix = np.matrix([row for row in make_matrix()])

        for emotion in filename_matrix.T:
            yield [self[img] for img in np.array(emotion)[0]]


if __name__ == '__main__':
    imdb = ImageDatabase(directory="/media/wolf/Shared/Dropbox/2016/QEA/2-Faces/faces")

    import matplotlib.pyplot as plt

    for emotion in imdb.iterate_emotions():
        for img in emotion:
            plt.imshow(img, cmap=plt.cm.gray)
            plt.show()
            break
        break

    for person in imdb.iterate_people():
        for emotion in person:
            plt.imshow(img, cmap=plt.cm.gray)
            plt.show()
            break
        break