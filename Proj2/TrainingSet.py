import DataSet
from DataSet import *
from typing import Any, Generic, Dict, List

class TrainingSet:
    def __init__(self):
        self.data = {}
        self.reduced = {}
        self.KCenters = {}
        self.banned = {}

    def isNoise(self, tolerance):
        pass

    def addDataSet(self, _data : DataSet):
        self.data[_data.getName()] = _data

    