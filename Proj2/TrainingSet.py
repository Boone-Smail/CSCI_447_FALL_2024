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
        self.banned : Dict[str, List[str]] = {}

    # helper function, designed to make a quick 
    # distance from 0^n accesible
    def distance(self, l : List[Any]):
        result = 0
        for i in l:
            result += i**2
        return result**0.5
    
    # helper function, returns distance between two points
    def distance_between(self, l1 : List[Any], l2 : List[Any]):
        result = 0
        for i in range(len(l1)):
            result += (l1[i] - l2[i]) ** 2
        return result**0.5

    # helper function for markNoise(...), it's meant to
    # just make sure the the contents of one list are in another
    def equal_content(self, l1, l2):
        for i in l1:
            if i in l2:
                l2.remove(i)
            else:
                return False
        return True

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
        for i in range(k):
            centroids.append(start + ((i/k)*stretch) + (0.5/k))
            accepted[i] = []
        # Do first k-means
            # Start by determining the distance from each centroid
        dist = []
        for i in fs.values:
            temp = [] # Calculate distance from each centroid for each point
            for j in centroids:
                temp.append(abs(i-j))
            dist.append(temp)

            # Assign each point a centroid
        for i in range(len(dist)):
            chosen = 0
            for j in range(len(dist[i])-1):
                if dist[i][chosen] > dist[i][j+1]:
                    chosen = j+1
            accepted[chosen].append(i)

            # Calculate new centroids
        new_centroids = []
        for i in accepted:
            temp = 0
            for j in accepted[i]:
                temp += fs.values[j]
            new_centroids.append(temp/len(accepted[i]))
        
        # Repeat k-means until centroids do not move
        while not self.equal_content(centroids, new_centroids):
            # reset centroids
            accepted = {}
            for i in range(k):
                accepted[i] = []
            centroids = new_centroids
            new_centroids = []

            # calculate distances from centroids from every point
            dist = []
            for i in fs.values:
                temp = []
                for j in centroids:
                    temp.append(abs(i-j))
                dist.append(temp)
            
            # classify points to centers
            for i in range(len(dist)):
                chosen = 0
                for j in range(len(dist[i])-1):
                    if dist[i][chosen] > dist[i][j+1]:
                        chosen = j+1
                accepted[chosen].append(i)
            
            # calculate new centroids
            for i in accepted:
                temp = 0
                for j in accepted[i]:
                    temp += fs.values[j]
                new_centroids.append(temp/len(accepted[i]))

            # Loop can now check if centroids have moved and repeat
        
        # With loop finished, centroids are placed.
        # Now to answer the question, are all centroids
        # of one disctinct class within the given
        # tolerance?
        suited = []
        for i in range(k):
            suited.append(i)
        for i in accepted:
            # check if all points of a cluster are within classification tolerance
            # and, if so, "suit" that classification.
            collected = {-1: 0}
            max_ind = -1
            total = 0
            for j in accepted[i]:
                if fs.classifications[j] in collected:
                    collected[fs.classifications[j]] += 1
                else:
                    collected[fs.classifications[j]] = 1
                if collected[fs.classifications[j]] > collected[max_ind]:
                    max_ind = fs.classifications[j]
                total += 1
            # With classifications counted, use the highest to calculate if it's within tolerance
            if (collected[max_ind]/total > tolerance):
                if max_ind in suited:
                    suited.remove(max_ind)
                else:
                    if fs.name not in self.banned[fs.dataset_name]: # Something has been suited twice! It is assumed making noise!
                        self.banned[fs.dataset_name].append(fs.name)
                    return True
            else:
                # Nothing found suitable within tolerance! This is noise!
                if fs.name not in self.banned[fs.dataset_name]:
                    self.banned[fs.dataset_name].append(fs.name)
                return True

    def addDataSet(self, _data : DataSet):
        self.data[_data.name] = _data
        self.banned[_data.name] = []

        classes = []
        for i in _data.classifications:
            if i not in classes:
                classes.append(i)
        k = len(classes)

        # print("Banned: ")
        # for i in _data.features:
        #     t = self.markNoise(FeatureSet.generateFeatureset(_data, i), k, 0.9)
        #     if(t):
        #         print("\t",i)

    def removeDataSet(self, _name : str):
        if _name not in self.reduced:
            self.banned.pop(_name)
        return self.data.pop(_name)
    
    def removeReduced(self, _name : str):
        if _name not in self.data:
            self.banned.pop(_name)
        return self.reduced.pop(_name)

    # Note that to classify an element in reduced, it cannot
    # exist in data
    def classify(self, name : str, k : int, element : List[Any]):
        if name in self.reduced:
            vote = {}
            dist = []
            index_ref = []
            count = 0
            # Find distances
            for i in self.reduced[name].data:
                index_ref.append(count)
                dist.append(self.distance_between(i, element))
                count += 1
            # sort distances
            for i in range(len(dist)):
                for j in range(len(dist) - (i+1)):
                    if i != j:
                        if dist[i] > dist[j+i]:
                            temp = dist[i]
                            dist[i] = dist[i+j]
                            dist[i+j] = temp

                            temp = index_ref[i]
                            index_ref[i] = index_ref[i+j]
                            index_ref[i+j] = temp
            max = 0
            max_ind = -1
            for i in range(k):
                chosen = self.reduced[name].classifications[index_ref[i]]
                if chosen in vote:
                    vote[chosen] += 1
                else:
                    vote[chosen] = 1
                if vote[chosen] > max:
                    max = vote[chosen]
                    max_ind = chosen
            # return result of plurality vote
            return max_ind
        elif name in self.data:
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
        else:
            return -1
    
    # Creates a reduced data set by only adding points
    # when n neighbors of the same classification are
    # within delta (_max_sd) range. Note that the reduced dataset
    # goes into 'reduced'. Also note that _max_sd plays on the standard
    # deviation of the dataset, and is intended to be used in a 0.5-3.5 range.
    # This function is destructive, and will overwrite any data
    # in reduced.
    def reduce(self, _name : str, _min_neighbors_percent : float, _max_sd : float):
        if _name in self.data:
            # Start by sorting the data
            sort = {}
            for i in range(len(self.data[_name].data)):
                c = self.data[_name].classifications[i]
                if c in sort:
                    sort[c].append(i)
                else:
                    sort[c] = [i] 

            # Then, take the sorted data and find the standard deviation of each
            # classification
            sd = []
            for i in sort: # For each classification
                # find the average (mean) length
                avg = 0
                for j in sort[i]: # For each element
                    temp = self.distance(self.data[_name].data[j])
                    avg += temp
                avg /= len(sort[i])
                # find the squared sums
                squared_sum = 0
                for j in sort[i]:
                    temp = (self.distance(self.data[_name].data[j]) - avg)
                    squared_sum += temp**2
                # Put standard deviation in sd
                sd.append(squared_sum/((len(sort[i])-1) if len(sort) > 1 else 1))
            # Then find the distance from each point
            # and mapping each point to what's in their range
            # (only including those of the same class)
            accepted = {}
            for i in sort:
                # start by finding the points within acceptable
                # distance to other points
                accepted[i] = {}
                for j in range(len(sort[i])):
                    p1 = self.data[_name].data[sort[i][j]]
                    for k in range(len(sort[i])-(j+1)):
                        if j != k:
                            p2 = self.data[_name].data[sort[i][j+k]]
                            if (self.distance_between(p1, p2) <= _max_sd*sd[i]):
                                if sort[i][j] in accepted[i]:
                                    accepted[i][sort[i][j]].append(j+k)
                                else:
                                    accepted[i][sort[i][j]] = [j+k]

            # finish by building the dataset
            # using only acceptable points
            # (edited k-nearest neighbor reduction)
            temp_data = []
            temp_classifications = []
            for i in accepted: # for each classification in accepted
                for j in accepted[i]: # for each mapping in the classification
                    if len(accepted[i][j]) >= len(accepted[i])*_min_neighbors_percent: # If this point has at least min_neighbors (in range, done earlier)
                        temp_data.append(self.data[_name].data[j])
                        temp_classifications.append(i)
                    elif (len(accepted[i]) == 1):
                        temp_data.append(self.data[_name].data[j])
                        temp_classifications.append(i)
            # build data set, and store in the right place
            d = DataSet(_name, temp_data, self.data[_name].features, self.data[_name].classes, temp_classifications, self.data[_name].discrete_reference)
            self.reduced[_name] = d
            return d
