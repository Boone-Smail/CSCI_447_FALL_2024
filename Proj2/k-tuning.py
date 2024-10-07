import DataSet, TrainingSet
from DataSet import *
from typing import Generic, Any, List, Dict
from ucimlrepo import fetch_ucirepo

if __name__=="__main__":
    # data = fetch_ucirepo(id=14)
    data = fetch_ucirepo(id=90)

    temp = DataSet.generate(data["data"]["original"], "Soy Bean")
    for i in data["data"]['original']:
        print(i)
    print(temp.getName())
    print(str(temp.getData()[0]))
    
