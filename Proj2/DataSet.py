from typing import Dict, List, Generic, Any
import random

class DataSet:
    def __init__(self, _name : str, _data : List[List[Any]], _features : List[str], _classes : List[str], _classifications : List[int]):
        self.name = _name
        self.data = _data
        self.features = _features
        self.classes = _classes
        self.classifications = _classifications
        self.size = len(self.data)

    def getName(self):
        return self.name
    
    def getData(self):
        return self.data
    
    def GetFeatures(self):
        return self.features
    
    @staticmethod
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
            temp_classes.append(i)
        temp_classifications = []
        temp_data = []
        for i in range(len(_data[classification_name])):
            temp = []
            for j in temp_features:
                temp.append(_data[j][i])
            if None not in temp:
                temp_data.append(temp)
                temp_classifications.append(temp_classes.index(_data[classification_name][i]))
        
        return DataSet(_name, temp_data, temp_features, temp_classes, temp_classifications)

    def randomStratified(int):
        pass