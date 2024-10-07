from ucimlrepo import fetch_ucirepo

print() # for whitespace

# Fetch Data
data = fetch_ucirepo(id=53)

# instantiate data structures
class_split = {}
class_total = 0
class_split["extracted"] = {}
for i in data["data"]["original"]["class"]:
    if i not in class_split["extracted"]:
        class_split["extracted"][i] = {}
for i in data["data"]["features"]:
    if i not in class_split["extracted"][data["data"]["original"]["class"][0]]:
        for j in class_split["extracted"]:
            class_split["extracted"][j][i] = {}

# Count area each class takes in the data set
for i in data["data"]["original"]["class"]:
    if i in class_split:
        class_split[i]["size"] += 1
    else:
        class_split[i] = {}
        class_split[i]["size"] = 1
    class_total += 1

# Separate data
count = 0
while (count < len(data["data"]["original"]["sepal length"])):
    item = []
    item.append(data["data"]["original"]["sepal length"][count])
    item.append(data["data"]["original"]["sepal width"][count])
    item.append(data["data"]["original"]["petal length"][count])
    item.append(data["data"]["original"]["petal width"][count])    
    item.append(data["data"]["original"]["class"][count])

    if "items" in class_split[item[4]]:
        class_split[item[4]]["items"].append(item)
    else:
        class_split[item[4]]["items"] = [item]

    count += 1

# Calculate formula on each attribute in each class
for i in class_split:
    if i != "extracted":        
        # start by counting attribute value occurances
        for j in class_split[i]["items"]:
            temp = data["data"]["features"].keys()
            for k in range(len(temp)):
                if j[k] in class_split["extracted"][i][temp[k]]:
                    class_split["extracted"][i][temp[k]][j[k]] += 1
                else:
                    class_split["extracted"][i][temp[k]][j[k]] = 1
        # Calculate the value found by the formula for each unique value in each feature in each class
        for j in class_split["extracted"][i]:
            for k in class_split["extracted"][i][j]:
                class_split["extracted"][i][j][k] = (k+1)/(len(class_split[i]["items"])+len(data["data"]["features"].keys()))

# Very simple test, take the first element from the second class and see if the algorithm classifies it correctly
test_sample = class_split["Iris-versicolor"]["items"][0] # << first item in the second class
likeliness = [] # << A list of likliness of the example belonging to the class, used to calculate argmax.
# Note that the class of the test_sample is Iris-versicolor. This becomes important at the very bottom.

# This is implementation of the example test given by the project outline
for i in class_split:
    if i != "extracted":
        features = list(class_split["extracted"][i].keys())
        # Start with the space the class takes in the data set
        qc = class_split[i]["size"]/class_total
        # and multiply it with the product of each attribute value's formula result (using associative property)
        for j in range(len(features)):
            try:
                # print(class_split["extracted"][i][features[j]]) # << This is debug
                qc *= class_split["extracted"][i][features[j]][test_sample[j]]
            except:
                # The attribute value does not occur, the element is NOT likely in this class; therefor *= 0
                qc *= 0
        # likeliness is calculated for the class, so store it
        likeliness.append(qc)

# find argmax
max = float(0)
max_ind = -1
for i in range(len(likeliness)):
    if likeliness[i] > max:
        max = likeliness[i]
        max_ind = i

# With argmax found, provide the suggested class
print("found classification:", list(class_split["extracted"].keys())[max_ind], "("+str(max_ind)+")")
# If this test (with the current code) returns "Iris-versicolor (1)", the training set is working correctly!