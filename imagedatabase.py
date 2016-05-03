#!/usr/bin/env python2

import os
import glob

import numpy as np
import scipy.misc


class ImageIterator(object):
    """
    Simple iterator class.
    """
    def __init__(self, generator, length):
        super(ImageIterator, self).__init__()
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator
        
        


class ImageDatabase(object):
    """
    ImageDatabase

    Provides helper functionality for dealing with a folder of images.

    Assumptions about the images in the directory:

    - There are an equal number of images of each person
    - The names of the images are of the format "x_name_magnitude.type"

    """


    def __init__(self, directory='face_images', image_filetype='png', use_grayscale=True, caching=True, resample=None):
        super(ImageDatabase, self).__init__()

        # Defines the image filetype extension to load
        self.image_filetype = image_filetype

        # Convert the image to grayscale on load?
        self.use_grayscale = use_grayscale

        # Cache images in a dictionary on load or reload every time?
        self.image_caching = caching

        # Resample images to (width, height) as defined by resample
        # None keeps image as is.
        self.resample = resample

        self.img_size = resample if resample != None else (360, 256)

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
        img = scipy.misc.imread(
            os.path.join(self.directory, name),
            self.use_grayscale
        )

        if self.resample != None:
            img = scipy.misc.imresize(img, self.resample)

        return img


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


    def _iterate_people_helper(self, with_names=None):
        """
        Yields the list of split filenames for each image for each person in 
        the image database.

        For use in the iterate_people method and iterate_emotions method.
        """
        split_f_names = self._get_split_filenames()

        people = sorted(list(self._get_people_set()))

        for person in people:
            if with_names == None or person in with_names:
                images_of_person = filter(
                    lambda split_f_name: split_f_name[self.name_idx] == person,
                    split_f_names
                )

                yield images_of_person


    def people(self, with_names=None):
        """
        Iterate the images by person. Yields an array of images for each person.
        """

        people_generator = ([self[self.split_char.join(split)] for split in images] for images in self._iterate_people_helper(with_names=with_names))

        return ImageIterator(people_generator, len(self._get_people_set()))

    def nth_of_person(self, index):
        return [imgs[index] for imgs in self.people()]


    def emotions(self):
        """
        Iterate the images by emotion. Yields an array of images for each emotion.
        """
        def make_matrix():
            for split_f_name_list in self._iterate_people_helper():
                yield [self.split_char.join(split_f_name) for split_f_name in split_f_name_list]

        # [names x emotions]
        filename_matrix = np.matrix([row for row in make_matrix()])

        emotion_generator = ([self[img] for img in np.array(emotion)[0]] for emotion in filename_matrix.T)

        return ImageIterator(emotion_generator, filename_matrix.shape[1])


    def subset(self, include=None, remove=None):
        """
        Default iteration method. Yields each image individually.
        """
        if include:
            filtered_list = [x for x in self.image_list if include in x]
        elif remove:
            filtered_list = [x for x in self.image_list if remove not in x]
        else:
            filtered_list = [x for x in self.image_list]

        subset_generator = (self[img] for img in filtered_list)

        return ImageIterator(subset_generator, len(filtered_list))



if __name__ == '__main__':
    imdb = ImageDatabase(directory="database")

    import matplotlib.pyplot as plt

    for emotion in imdb.emotions():
        for img in emotion:
            plt.imshow(img, cmap=plt.cm.gray)
            plt.show()
            break
        break

    for person in imdb.people():
        for emotion in person:
            plt.imshow(img, cmap=plt.cm.gray)
            plt.show()
            break
        break
