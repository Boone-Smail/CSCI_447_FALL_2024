import DataSet
from typing import Generic, List, Dict, Any

class FeatureSet:
    def __init__(self, _dataset_name : str, _name : str, _values : List[int], _classifications : List[int]):
        self.dataset_name : str = _dataset_name
        self.name : str = _name
        self.values : List[int] = _values
        self.classifications : List[int] = _classifications

    @staticmethod
    def generateFeatureset(data, feature_name:str):
        temp_vals = data.extractFeature(feature_name)
        return FeatureSet(data.name, feature_name, temp_vals, data.classifications)
    
