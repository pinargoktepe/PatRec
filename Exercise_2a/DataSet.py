import numpy as np


class DataSet(object):

    def __init__(self, filename: str):
        """

        :type filename: str
        """
        self.filename = filename
        array = np.loadtxt(filename, delimiter=",", dtype="int")
        self.labels = array[:, 0]
        self.data = array[:, 1:]
