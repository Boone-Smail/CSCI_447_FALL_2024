import DataSet, TrainingSet, FeatureSet
from TrainingSet import *
from FeatureSet import *
from DataSet import *
from typing import Generic, Any, List, Dict
from ucimlrepo import fetch_ucirepo

def ten_fold_test(data : DataSet, k : int):
    t = TrainingSet()
    t.addDataSet(data)
    stratified = data.stratified(10)
    acc = []
    for i in range(len(stratified)):
        recovered = [] 
        temp = []
        for j in stratified[i]:
            recovered.append(data.remove(j))
        for j in range(len(stratified[i])):
            chosen = t.classify(data.name, k, stratified[i][j])
            if chosen == recovered[j]:
                temp.append(1)
            else:
                temp.append(0)
        acc.append(sum(temp)/len(temp))
        for j in range(len(stratified[i])):
            data.add(stratified[i][j], recovered[j])
    return sum(acc)/len(acc)


# This is mostly a helper function; it's
# meant to only be used in k_tune. Returns
# average accuracy over the folds in 'stratified'
# using a given k for the training set to test
# on.
def _single_k_tune(data: DataSet, stratified : List[List[Any]], test_set : List[List[Any]], test_set_classes : List[int], k : int):
    accuracies = []

    # output = "[ "
    # for i in stratified:
    #     output += "{:.2f}".format(len(i)/len(data.data)) + ", "
    # print(output.rstrip(", ") + " ]")

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
    return 0 if len(accuracies) == 0 else sum(accuracies)/len(accuracies)
        

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
        if (acc[-1] > acc[max_ind]): # Check for new max
            max_ind = len(acc)-1 
        k += 1
        print("===========\nK: " + str(k-1) + "\t Accuracy: " + "{:.3f}".format(acc[-1]))
    
    return max_ind # Diminishing returns found, return k with max accuracy 
    
# Finds best standard deviations for the provided min_neighbors
def reduce_single(data :DataSet, test_set : List[List[Any]], test_set_classifications : List[int], k : int, min_neighbors : int):
    t = TrainingSet()
    t.addDataSet(data)
    acc = []
    for i in range(7):
        temp = []
        t.reduce(data.name, min_neighbors, ((0.5*i)+1))
        print("classifying...\t", len(test_set)) 
        for j in range(len(test_set)):
            if t.classify(data.name, k, test_set[j]) == test_set_classifications[j]:
                temp.append(1)
            else:
                temp.append(0)
        acc.append((sum(temp)/len(temp), (0.5*i)+1))
    # return most accurate sd
    max = -1
    sd = 1
    for i in acc:
        if i[0] > max:
            max = i[0]
            sd = i[1]
    return (sd)
            

# A function to tune how finely the data is reduced
# so that the best reduction's standard deviation is found
def tune_reduction_k(data : DataSet, test_set : List[List[Any]], test_set_classifications : List[int], k : int):
    # find best standard deviation     
    print("Doing reductions...")
    sd = reduce_single(data, test_set, test_set_classifications, k, 0.1)

    # Use best standard deviation value to reduce and tune
    print("Tunining k...")
    t = TrainingSet()
    t.addDataSet(data)
    reduced = t.reduce(data.name, 0.1, sd)

    # now do k-tuning
    return (sd, k_tune(reduced, test_set, test_set_classifications, k))

if __name__=="__main__":
    # First, fetch all data
    data = {}
    
    # Commented out for time constraint; ran out of time
    # to solve any regression so all regression problems
    # and related entities are removed
    # data["abalone"] = fetch_ucirepo(id=1)
    # data["hardware"] = fetch_ucirepo(id=29)
    # data["fires"] = fetch_ucirepo(id=162)
    data["glass"] = fetch_ucirepo(id=42)
    data["breast cancer"] = fetch_ucirepo(id=14)
    data["soybean"] = fetch_ucirepo(id=90)

    regression = ['abalone', 'hardware', 'fires']

    k_test = {}
    edited_test = {}

    # Then, tune k for each dataset
    for n in data:
        temp = DataSet.generate(data[n]["data"]["original"], n)
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
        for j in x[1]:
            output += "=="
        print(output)
        
        # t = TrainingSet()
        # t.addDataSet(temp)
        # c = t.classify(temp.name, 10, x[0][9])
        # print(c)

        # k_test = k_tune(temp, x[0], x[2], 7)
        # print("Best k: " + str(k_test + 1) + "\n=-=-=-=-=-=-=-=-=-=-=\n")

        
        edited_test[n] = (tune_reduction_k(temp, x[0], x[2], 7))
        print("Best k [edited]:", edited_test[n][1] + 1,"\n=-=-=-=-=-=-=-=-=-=\n")
    
        k_test[n] = (k_tune(temp, x[0], x[2], 5))
        print("Best k [nearest]:", k_test[n])

    print()
    for i in edited_test:
        print(str(i) + ":")
        print("\t", edited_test[i][1] + 1)
        print("\t", k_test[i] + 1)

    print("\n\nTen-fold test results:")
    for i in data:
        print("\n", i)
        temp = DataSet.generate(data[i]["data"]["original"], i)
        print("\tnearest:\t", "{:.3f}".format(ten_fold_test(temp, k_test[i])))

        t = TrainingSet()
        t.addDataSet(temp)
        r = t.reduce(temp.name, 0.1, edited_test[i][0])
        print("\tedited:\t\t", "{:.2f}".format(ten_fold_test(r, edited_test[i][1])))