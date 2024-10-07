from ucimlrepo import fetch_ucirepo

print("\n") # for whitespace

# fetch data
data = {}
data["Breast Cancer"] = fetch_ucirepo(id=17) # << Will need try catch
data["Glass"] = fetch_ucirepo(id=42) # << Will need try catch
data["Iris"] = fetch_ucirepo(id=53)
data["Soybean"] = fetch_ucirepo(id=91)
data["Vote"] = fetch_ucirepo(id=105) # << will need try catch

#print(list(data["Breast Cancer"]["data"]["targets"].keys()))

# instantiate structures that training set is built of (refer to doc)
training_set = {}
for i in data:
    training_set[i] = {}
    training_set[i]["base"] = {}
    training_set[i]["extracted"] = {}
    features = list(data[i]["data"]["features"].keys())
    for j in data[i]["data"]["targets"]:
        for class_name in data[i]["data"]["targets"][j]:
            if class_name not in training_set[i]["base"]:
                training_set[i]["base"][class_name] = []
                training_set[i]["extracted"][class_name] = {}
                for k in features:
                    training_set[i]["extracted"][class_name][k] = {}

                training_set[i]["extracted"][class_name]["size"] = 0
            for k in data[i]["data"]["features"]:
                training_set[i]["extracted"][class_name][k] = {}

# Debug, these will run without error and
# show the correct clasification names when
# the setup above is done correctly
# print(list(training_set["Breast Cancer"]["base"].keys()))
# print(list(training_set["Glass"]["base"].keys()))
# print(list(training_set["Iris"]["base"].keys()))
# print(list(training_set["Soybean"]["base"].keys()))
# print(list(training_set["Vote"]["base"].keys()))

#print(list(data["Iris"]["data"]["original"]["sepal length"])[0])

# Extract necesary data
# Starting with presence of class in dataset
# and the sorting of data
for i in data:
    total = 0
    features = list(data[i]["data"]["features"].keys())
    for j in range(len(data[i]["data"]["features"][features[0]])):
        temp = []
        for k in features:
            temp.append(data[i]["data"]["features"][k][j])

        if None not in temp:
            class_type = (data[i]["data"]["targets"].keys())[0]
            training_set[i]["base"][data[i]["data"]["targets"][class_type][j]].append(temp)
            total += 1
    for j in training_set[i]["base"]:
        training_set[i]["extracted"][j]["size"] = len(training_set[i]["base"][j])/total

print(training_set["Iris"]["extracted"]["Iris-setosa"]["size"])

# Then calculate the formula from #3 in the outline
for i in data:
    for j in training_set[i]["base"]:
        features = list(training_set[i]["extracted"][j].keys())
        features.remove("size")
        for k in range(len(features)):
            # count value occurances
            for l in range(len(training_set[i]["base"][j])):
                cur_val = training_set[i]["base"][j][l][k]
                if (cur_val not in training_set[i]["extracted"][j][(features[k])].keys()):
                    training_set[i]["extracted"][j][features[k]][cur_val] = 1
                else:
                    training_set[i]["extracted"][j][features[k]][cur_val] += 1
            # run values through formula
            for l in training_set[i]["extracted"][j][features[k]]:
                training_set[i]["extracted"][j][features[k]][l] += 1
                training_set[i]["extracted"][j][features[k]][l] /= (len(training_set[i]["base"][j])+len(features))
            training_set[i]["extracted"][j][features[k]]["default"] = 1/(len(training_set[i]["base"][j])+len(features))
