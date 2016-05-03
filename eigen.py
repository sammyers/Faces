from faces import FaceSpace
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class EigenSpace(FaceSpace):
    """
    A face space using the Eigenfaces method.
    """
    def __init__(self, images, dimensions=(360, 256)):
        super(EigenSpace, self).__init__(images, dimensions)

        self.eigenvectors = self.get_eigenvectors()

    def get_eigenvectors(self, limit=None):
        """
        Return a list of column vectors representing the eigenfaces of a set of images.
        """
        eigenthings = np.linalg.eig(self.get_covariance_matrix(self.image_vectors[::8]))
        vectors = self.normalize_faces(self.image_vectors[::8]) * eigenthings[1]
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
    all_faces = EigenSpace(imdb.subset(add='01'), imdb.img_size)

    # for emotion in imdb.iterate_emotions():
    #     e_faces.show_eigenfaces(emotion)
    # all_faces.show_eigenfaces()
    jared = imdb['faceimage_jaredBriskman_06.png']

    return all_faces.recognize_face(jared)

def make_emotion_eigenspace(imdb):
    eigenemotions = [EigenSpace(emotion) for emotion in imdb.emotions()]

    return EigenSpace([np.reshape(e_emotion.get_eigenvectors(limit=1), imdb.img_size) for e_emotion in eigenemotions])


if __name__ == '__main__':
    from imagedatabase import ImageDatabase

    imdb = ImageDatabase(directory="database", resample=(90,64))

    # final_faces = ImageDatabase(directory="finalFaces")

    # e_space = EigenSpace(imdb.subset(remove='01'))
    e_space = EigenSpace(imdb)

    print('EigenSpace created')

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

    # targets = list(imdb.subset(include='01'))
    targets = final_faces

    score = 0
    matched = []

    for i, target in enumerate(targets):

        index, image, e_dist = e_space.recognize_face(target)

        if i * 8 <= index < (i + 1) * 8:
            score += 1
        matched += [target, image]

        # e_space.show_eigenfaces()

        # plt.imshow(image, cmap=plt.cm.gray)
        # plt.show()    

    print('{}/{}'.format(score, len(targets)))
    e_space.show_grid(matched)
