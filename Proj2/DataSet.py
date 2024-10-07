from typing import Dict, List, Generic, Any
import numpy as np
from numpy import isnan
import random

class DataSet:
    def __init__(self, _name : str, _data : List[List[Any]], _features : List[str], _classes : List[str], _classifications : List[int]):
        self.name = _name
        self.data = _data
        self.features = _features
        self.classes = _classes
        self.classifications = _classifications
        self.size = len(self.data)
    
    def generate(_data : Dict[str,List[any]], _name : str):
        element_length = 0
        classification_name = ""
        temp_features = []
        for i in _data:
            if "class" not in i.lower():
                temp_features.append(i)
                element_length += 1
            else:
                classification_name = i
        temp_classes = []
        for i in _data[classification_name]:
            if i not in temp_classes:
                temp_classes.append(i)
        temp_classifications = []
        temp_data = []
        for i in range(len(_data[classification_name])):
            temp = []
            for j in temp_features:
                example = _data[j][i]
                temp.append(example)
            nanPresent = False
            for j in temp:
                if np.isnan(j):
                    nanPresent = True
                    break
            if None not in temp and not nanPresent:
                temp_data.append(temp)
                temp_classifications.append(temp_classes.index(_data[classification_name][i]))
        
        return DataSet(_name, temp_data, temp_features, temp_classes, temp_classifications)

    def extractFeature(self, feature:str):
        if (feature in self.features):
            index = self.features.index(feature)
            temp = []
            for i in self.data:
                temp.append(i[index])
            return temp
        return []

    # Given a percentage as a float (60% = 0.6), 
    # returns a stratified sample of the database
    def randomStratified(self, percent:float):
        class_sort = {}
        for i in self.classes:
            class_sort[i] = []
        for i in range(len(self.classifications)):
            class_sort[self.classes[self.classifications[i]]].append(self.data[i])
        temp = []
        presence = []
        temp_presence = []
        for i in class_sort:
            presence.append(len(class_sort[i])/len(self.data))
            temp_presence.append(1)
        
        # start with at least 1 example from each classification
        for i in class_sort:
            if len(class_sort[i]) > 0:
                temp.append(class_sort[i].pop(random.randint(1,1000)%len(class_sort[i])))

        while (len(temp)/len(self.data) < percent):
            chosen = 0
            if (len(temp) > 0):
                # Had an idea for a statistically based stratification idea,
                # where after collecting 1 sample from each class you take
                # a sample from the most likely class (povided you already
                # know what you've added to the stratified samples). This is so
                # you can build one that's n% the size of the original that's
                # equally stratified while remaining roughly the same representation
                # as the original
                selector_idea = []
                for i in range(len(presence)):
                    selector_idea.append((presence[i] - (temp_presence[i]/len(temp))))
                chosen = selector_idea.index(max(selector_idea))
            temp.append(class_sort[self.classes[chosen]].pop(random.randint(0,1000)%(len(class_sort[self.classes[chosen]]))))
            temp_presence[chosen] += 1

        # return stratified examples
        return (temp, temp_presence)
    
    def remove(self, element):
        if len(element) == len(self.data[0]):
            self.data.remove(element)
        else:
            return False