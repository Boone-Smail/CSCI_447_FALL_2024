from typing import Dict, List, Generic, Any
import numpy as np
from numpy import isnan
import random
import array

class DataSet:
    def __init__(self, _name : str, _data : List[List[Any]], _features : List[str], _classes : List[str], _classifications : List[int], _dr : Dict[str,Any]):
        self.name = _name
        self.data = _data
        self.features = _features
        self.classes = _classes
        self.classifications = _classifications
        self.size = len(self.data)

        self.discrete_reference = _dr
    
    def generate(_data : Dict[str,List[any]], _name : str):
        element_length = 0
        classification_name = ""
        temp_features = []
        temp_dr = {} # << categorical var reference table
        for i in _data:
            if "class" not in i.lower() and "type_of" not in i.lower() and "rings" != i.lower() and ("area" == i.lower() and _name != "hardware"):
                temp_entry = i
                temp_features.append(i)
                element_length += 1
            else:
                classification_name = i
        temp_classes = []
        for i in _data[classification_name]:
            print(i)
            if i not in temp_classes and not np.isnan(i):
                temp_classes.append(i)
        temp_classifications = []
        temp_data = []
        for i in range(len(_data[classification_name])):
            temp = []
            for j in temp_features:
                example = _data[j][i]
                temp.append(example)
                for j in range(len(temp)): # Check for discrete variables
                    if isinstance(temp[j], str):
                        cat_var = temp[j]
                        if temp_features[j] not in temp_dr:
                            temp_dr[temp_features[j]] = {"spot" : 0}
                        if cat_var in temp_dr[temp_features[j]]:
                            temp[j] = temp_dr[temp_features[j]][cat_var]# << Assign one-hot encoding
                        else:
                            temp_dr[temp_features[j]][cat_var] = int(temp_dr[temp_features[j]]["spot"])
                            temp[j] = int(temp_dr[temp_features[j]]["spot"]) # New categorical value found, add it and increment "spot"
                            temp_dr[temp_features[j]]["spot"] += 1

            nanPresent = False
            for j in temp:
                if np.isnan(j):
                    nanPresent = True
                    break

            if np.isnan(_data[classification_name][i]):
                nanPresent = True

            if None not in temp and not nanPresent:
                print(_data[classification_name][i])
                
                temp_data.append(temp)
                temp_classifications.append(temp_classes.index(_data[classification_name][i]))
        
        return DataSet(_name, temp_data, temp_features, temp_classes, temp_classifications, temp_dr)

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
        temp_order = []
        for i in class_sort:
            presence.append(len(class_sort[i])/len(self.data))
            temp_presence.append(1)
        
        # start with at least 1 example from each classification
        for i in class_sort:
            if len(class_sort[i]) > 0:
                temp.append(class_sort[i].pop(random.randint(1,1000)%len(class_sort[i])))
                temp_order.append(i)

        # Now fill with examples until the stratified example matches or exceeds size        
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
            temp_order.append(chosen)

        # return stratified examples
        return (temp, temp_presence, temp_order)
    
    def remove(self, element):
        if len(element) == len(self.data[0]):
            index = self.data.index(element)
            self.data.pop(index)
            c = self.classifications.pop(index)
            return c 
        else:
            return False
        
    def  add(self, element : List[Any], classification : int):
        if len(element) == len(self.data[0]):
            for j in range(len(element)):
                if type(element[j]) != type(self.data[0][j]):
                    pass
                    # return False
            self.data.append(element)
            self.classifications.append(classification)
            return True
        return False
    
    def stratified(self, folds : int):
        split = []
        while len(split) < folds:
            temp = random.random()
            if temp > 0.03 and temp < 0.97: # Just making sure no folds are too small
                passes = True
                for i in split:
                    if abs(temp-i) < 0.04:
                        passes = False
                        break
                if passes:
                    split.append(temp)

        split = sorted(split)
        split.append(1)
        
        # Now sort the data
        sort : Dict[int, List[int]] = {}
        for i in range(len(self.data)):
            if self.classifications[i] in sort:
                sort[self.classifications[i]].append(i)
            else:
                sort[self.classifications[i]] = [i]

        # And put the sorted data into their stratified folds
        spot = 0
        subspot = 0
        available = []
        for i in sort:
            if len(sort[i]) > 0:
                available.append(i)
        ret = []
        temp = []
        for i in range(len(self.data)):
            if i <= split[spot]*len(self.data):
                temp.append(self.data[sort[available[subspot]][0]])
                sort[available[subspot]].pop(0)
            else:
                ret.append(temp)
                temp = []
                temp.append(self.data[sort[available[subspot]][0]])
                sort[available[subspot]].pop(0)
                spot += 1 # << Move spot so the next item goes to the next "fold"
            # Move subspot so element from next
            # classification is selected
            if len(sort[available[subspot]]) == 0:
                available.pop(subspot)
            else:
                subspot += 1
            if (subspot >= len(available)):
                subspot = 0
        return ret