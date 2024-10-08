import DataSet
import FeatureSet
from FeatureSet import *
from DataSet import *
from typing import Any, Generic, Dict, List

class TrainingSet:
    def __init__(self):
        self.data : Dict[str, DataSet] = {}
        self.reduced : Dict[str, DataSet] = {}
        self.KCenters = {}
        self.banned = {}

    # implements k-means to determine
    # if cluster centroids are too close to
    # each other in this feature
    def markNoise(self, fs : FeatureSet, k : int, tolerance : float):
        # Set up structures for k-means
        accepted = {}
        centroids = []
        # Start with arbitrary start values for centroid positions
        start = min(fs.values)
        stretch = max(fs.values)-start
        for i in k:
            centroids.append(start + ((i/k)*stretch) + (0.5/k))
        # Do first k-means

    def addDataSet(self, _data : DataSet):
        self.data[_data.name] = _data

    def removeDataSet(self, _name : str):
        return self.data.pop(_name)

    def classify(self, name : str, k : int, element : List[Any]):
        if name in self.data:
            if len(element) == len(self.data[name].data[0]):
                vote = {}
                distances = []
                dist_indexes = []
                sum = 0
                # calculate distances
                for i in range(len(self.data[name].data)):
                    for j in range(len(self.data[name].data[i])):
                        sum += (element[j]-self.data[name].data[i][j])**2
                    distances.append(sum**0.5)
                    dist_indexes.append(i)
                    sum = 0
                # sort distances
                for i in range(len(distances)):
                    for j in range(len(distances)-i):
                        if i != j:
                            if distances[i] > distances[j + i]:
                                temp = distances[i]
                                distances[i] = distances[j + i]
                                distances[j + i] = temp

                                temp = dist_indexes[i]
                                dist_indexes[i] = dist_indexes[i + j]
                                dist_indexes[j + i] = temp
                # Select k nearest neighbors for vote
                max = 0
                winner = 0
                for i in range(k):
                    candidate = self.data[name].classifications[dist_indexes[i]]
                    if candidate in vote:
                        vote[candidate] += 1
                    else:
                        vote[candidate] = 1
                    if vote[candidate] > max:
                        max = vote[candidate]
                        # Reasoning here is that if two candidates
                        # have the same neighbor counts, the group
                        # with the closest neighbors (reaching max
                        # first via the sorting) wins the election
                        winner = candidate
                return winner
                
            else:
                return -2
        elif name in self.reduced:
            return 0
        else:
            return -1