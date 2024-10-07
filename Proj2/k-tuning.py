import DataSet, TrainingSet
from DataSet import *
from typing import Generic, Any, List, Dict
from ucimlrepo import fetch_ucirepo

if __name__=="__main__":
    # data = fetch_ucirepo(id=14)
    data = fetch_ucirepo(id=90)

    temp = DataSet.generate(data["data"]["original"], "Soy Bean")
    # for i in data["data"]['original']:
    #     print(i)
    
    x = temp.randomStratified(0.1)
    
    
    print("Results:\n" + str(len(x[0])) + " =? " + " {:.2f}".format(float(len(temp.data))*0.1))
    output = ""
    for i in range(x[1][x[1].index(max(x[1]))]):
        tempout = ""
        for j in x[1]:
            if j > i:
                tempout += "X "
            else:
                tempout += "  "
        tempout += "\n"
        output = tempout + output
    print(output)
