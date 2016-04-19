#!/usr/bin/env python2

import os
import glob

import numpy as np
import scipy

class ImageDatabase(object):
    """
    ImageDatabase

    Provides helper functionality for dealing with a folder of images
    """
    image_filetype = "png"
    use_grayscale = True
    image_caching = True

    split_char = "_"

    name_idx = 1
    magnitude_idx = 2


    def __init__(self, directory='face_images'):
        super(ImageDatabase, self).__init__()

        if not _check_valid_directory(directory):
            raise Exception("Invalid directory")

        self.directory = directory

        self.image_list = sorted(_get_image_list(self.directory))

        self.cached_images = {}


    def _get_image_list(directory):
        """
        Get all images of the right type in directory
        """
        glob_path = os.path.join(directory, "*.{}".format(image_filetype))
        return glob.glob(glob_path)


    def _images_in_directory(directory):
        # are there more than 0 images of the right type in this directory?
        return len(_get_image_list(directory)) > 0


    def _check_valid_directory(directory):
        return (os.path.isdir(directory) and
            _images_in_directory(directory))


    def _load_image(self, name):
        return scipy.ndimage.imread(
            os.path.join(self.directory, name),
            use_grayscale
        )


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, key):
        if image_caching:
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
        for img in self.image_list:
            yield self[img]


    def _get_split_filenames(self):
        return [f_name.split(split_char) for f_name in self.image_list]


    def _get_people_set(self):
        """
        Make some assumptions about the images in the directory:

        - There are going to be an equal number of images of each person
        - The names of the images are of the format "x_name_magnitude.type"
        """
        split_f_names = self._get_split_filenames()

        return set(split_f_names[:, name_idx])

    def _iterate_people_helper(self):
        """
        Yields the list of split filenames for each person in the image database,
        for use in the iterate_people method and iterate_emotions method
        """
        people = _get_people_set()

        for person in people:
            images_of_person = filter(
                lambda split_f_name: split_f_name[name_idx] is person,
                split_f_names
            )

            yield images_of_person

    def iterate_people(self):
        for images_of_person in self._iterate_people_helper():

            # rejoin split string and yield array of images of the person
            yield [self[f_name.join(split_char)] for f_name in images_of_person]


    def iterate_emotions(self):
        def make_matrix():
            for split_f_name_list in self._iterate_people_helper():
                yield [split_f_name.join(split_char) for split_f_name in split_f_name_list]

        # [names x emotions]
        filename_matrix = np.matrix([row for row in make_matrix()])

        for emotion in filename_matrix.T:
            yield [self[img] for img in emotion]
