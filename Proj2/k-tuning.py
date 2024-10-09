import DataSet, TrainingSet, FeatureSet
from TrainingSet import *
from FeatureSet import *
from DataSet import *
from typing import Generic, Any, List, Dict
from ucimlrepo import fetch_ucirepo

# This is mostly a helper function; it's
# meant to only be used in k_tune. Returns
# average accuracy over the folds in 'stratified'
# using a given k for the training set to test
# on.
def _single_k_tune(data: DataSet, stratified : List[List[Any]], test_set : List[List[Any]], test_set_classes : List[int], k : int):
    accuracies = []

    output = "[ "
    for i in stratified:
        output += "{:.2f}".format(len(i)/len(data.data)) + ", "
    print(output.rstrip(", ") + " ]")

    # For every fold
    for i in stratified:
        # Create initial data
        t = TrainingSet()
        recovered = []
            # This includes trimming down the data set for that 90% of 90%
        for j in i:
            recovered.append(data.remove(j)) # add classes of removed to 'recovered' for later
        #Now give the training set has that sweet 81% data it's "trained" on
        t.addDataSet(data)
        
        # guess on this fold
        temp_acc = []
        for j in range(len(test_set_classes)):
            guess = t.classify(data.name, k, test_set[j]) # << Gets guess from k-neighbors in TS.classify
            if guess == test_set_classes[j]: # Employ zero-one loss
                temp_acc.append(1)
            else:
                temp_acc.append(0)
        # Use average of zero-one loss as accuracy for this holdout
        accuracies.append((sum(temp_acc)/len(temp_acc)))

        # re-add the data to the dataset...
        for j in range(len(i)):
            data.add(i[j], recovered[j]) # ... using the recovered classes from earlier
    # return average accuracy
    return sum(accuracies)/len(accuracies)
        

def k_tune(data : DataSet, test_set : List[List[Any]], test_set_classifications : List[int], min_k = 3):
    st = data.stratified(10)
    acc = []
    # Find initial tuning of 1-k times
    for i in range(min_k):
        acc.append(_single_k_tune(data, st, test_set, test_set_classifications, i+1)) # << add accuracy of k-test to accuracies
        print("Accuracy: {:.3f}".format(acc[-1]))

    # While there are still less than min_k entries from the max accuracy,
    # keep increasing k
    max_ind = acc.index(max(acc))
    k = min_k + 1
    while (max_ind + min_k >= len(acc)):
        acc.append(_single_k_tune(data, st, test_set, test_set_classifications, k)) # Find accuracy of increased k
        if (acc[-1] >= acc[max_ind]): # Check for new max
            max_ind = len(acc)-1 
        k += 1
        print("===========\nK: " + str(k-1) + "\t Accuracy: " + "{:.3f}".format(acc[-1]))
    
    return max_ind # Diminishing returns found, return k with max accuracy 
    



if __name__=="__main__":
    # First, fetch all data
    data = {}
    data["breast cancer"] = fetch_ucirepo(id=14)
    data["soybean"] = fetch_ucirepo(id=90)

    # Then, tune k for each dataset
    for i in data:
        temp = DataSet.generate(data[i]["data"]["original"], i)
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
        print(output.rstrip())
        output = ""
        for i in x[1]:
            output += "=="
        print(output)
        
        # t = TrainingSet()
        # t.addDataSet(temp)
        # c = t.classify(temp.name, 10, x[0][9])
        # print(c)

        k_test = k_tune(temp, x[0], x[2], 20)
        print("Best k: " + str(k_test + 1) + "\n=-=-=-=-=-=-=-=-=-=-=\n")