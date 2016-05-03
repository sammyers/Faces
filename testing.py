import matplotlib.pyplot as plt
import numpy as np
import time

from imagedatabase import ImageDatabase
from eigen import EigenSpace

# plt.xkcd()


def timed(string):
    def timed_dec(f):
        def wrap(*args, **kwargs):
            time1 = time.time()
            ret = f(*args, **kwargs)
            time2 = time.time()
            print('%s in %0.3f ms' % (string, (time2-time1)*1000.0))
            return ret
        return wrap
    return timed_dec


class TestMethod(object):
    """TestMethod provides methods to test a facial recognition technique"""
    def __init__(self):
        super(TestMethod, self).__init__()
        self.name = "Unnamed test"

    def setup(self):
        pass

    def run_test(self):
        pass


class TwoDTestSuite(object):
    """docstring for TwoDTestSuite"""
    def __init__(self, test_name, test_methods):
        super(TwoDTestSuite, self).__init__()
        self.arg = arg
        
        


class TestSuite(object):
    """TestSuite runs accuracy tests on given test methods"""
    def __init__(self, test_name, test_methods, xlabel=None):
        super(TestSuite, self).__init__()
        self.test_name = test_name
        self.test_methods = test_methods
        self.results = []

        self.xlabel = xlabel

    def run_tests(self):
        for method in self.test_methods:
            method.setup()
            self.results.append(method.run_test())

    def bar_graph(self):
        data = [float(x[1]) / float(x[2]) * 100.0 for x in self.results]
        print data
        N = len(data)
        ind = np.arange(N)
        width = 0.35

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(ind, data, width)

        ax.set_title(self.test_name)

        ax.set_xlabel(self.xlabel or 'Technique')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(ind + width / 2)
        # ax.set_xticks([0, 1, 2])
        # ax.set_xlim([-0.5, 2.5])
        ax.set_ylim([0, 110])
        ax.set_xticklabels([t.name for t in self.test_methods])
        plt.show()


class EigenfacesTest(TestMethod):
    """docstring for EigenfacesTest"""
    def __init__(self, name, training_faces, test_faces):
        super(EigenfacesTest, self).__init__()
        
        self.name = name
        self.training_faces = training_faces
        self.test_faces = test_faces

    @timed("Eigenspace created from training faces")
    def setup(self):
        self.eigenspace = EigenSpace(self.training_faces)

    @timed("Test finished")
    def run_test(self):
        targets = self.test_faces

        score = 0
        matched = []

        for i, target in enumerate(targets):

            index, image, e_dist = self.eigenspace.recognize_face(target)

            print i, index

            if index == i / 7:
                score += 1

            matched.append(np.hstack((target,image)))

        print("Matched {}/{}".format(score, len(targets)))

        self.eigenspace.show_grid(matched)
        # self.eigenspace.show_grid(targets)
        # self.eigenspace.show_grid(self.training_faces)
        return (matched, score, len(targets))


@timed("get ff pairs")
def get_ff_pairs(imdb, ff, index):
    return [imgs[index] for imgs in imdb.people(with_names=ff._get_people_set())]

@timed("get nth image")
def get_nth_image(imdb, index, with_names=None):
    return [imgs[index] for imgs in imdb.people(with_names=with_names)]

@timed("get non nth image")
def get_non_nth_image(imdb, index, with_names=None):
    images = []
    for imgs in imdb.people(with_names=with_names):
        images += [img[1] for img in enumerate(imgs) if img[0] != index]
    return images
        
if __name__ == '__main__':
    from imagedatabase import ImageDatabase

    imdb = ImageDatabase(directory="database", resample=(90,64))

    ff = ImageDatabase(directory="finalFaces", resample=(90,64))

    no_faces = ImageDatabase(directory="no-faces", resample=(90,64))


    no_face_tests = [EigenfacesTest("{}".format(n), get_nth_image(no_faces, 0), get_nth_image(imdb, n)) for n in range(8)]
    no_face_test_suite = TestSuite("Face Recognition: Normal Set (n-th) vs. No-faces (1st)", no_face_tests, xlabel="n-th image from Normal set matched with Normal set minus the face image")
    no_face_test_suite.run_tests()
    no_face_test_suite.bar_graph()

    # ff_tests = [EigenfacesTest("{}".format(n), get_nth_image(imdb, n, with_names=ff._get_people_set()), get_nth_image(ff, 0, with_names=ff._get_people_set())) for n in range(8)]
    # ff_test_suite = TestSuite("Face Recognition: Normal Set (n-th) vs. finalFaces", ff_tests, xlabel="n-th image from Normal set matched with finalFaces image")
    # ff_test_suite.run_tests()
    # ff_test_suite.bar_graph()


    # one_minus_tests = []
    # for n in range(8):
    #     one_minus_tests.append(
    #         EigenfacesTest(
    #             "{}".format(n),
    #             get_non_nth_image(imdb, n),
    #             get_nth_image(imdb, n)
    #         )
    #     )

    # one_minus_test_suite = TestSuite("Face Recognition: Normal Set Minus n-th vs. Normal Set n-th", one_minus_tests, xlabel="Eigenspace created from every image besides n-th image from Normal set\nmatched with n-th image from Normal set")
    # one_minus_test_suite.run_tests()
    # one_minus_test_suite.bar_graph()



    # test_suite = TestSuite(ff_tests)

    # test_suite.run_tests()

    # test_suite.bar_graph()

