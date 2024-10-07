from ucimlrepo import fetch_ucirepo
import random
import scipy

print("\n") # for whitespace

# fetch data
data = {}
data["Breast Cancer"] = fetch_ucirepo(id=17) 
data["Glass"] = fetch_ucirepo(id=42) 
data["Iris"] = fetch_ucirepo(id=53)
data["Soybean"] = fetch_ucirepo(id=91)
data["Vote"] = fetch_ucirepo(id=105) 

# print(list(data["Iris"]["data"]["original"].keys()))

# create 10-fold splits randomly
split = []
for i in range(9):
    split.append(random.random())
if 1 in split:
    temp = split.append(random.random())
    while (temp == 1):
        temp = split.append(random.random())
    split.append(temp)
else:
    split.append(1)

split = sorted(split)

# debug for split
output = "[ "
for i in split:
    output += "{:.3f}".format(i) + ", "
output = output.rstrip(", ") + " ]"
# print(output)

accuracies = []
print("Generating results...")
# Then test them by holding out the current range, and creating the test set on the rest
for cur_spot in range(10):
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

    # Extract necesary data
    # Starting with presence of class in dataset
    # and the sorting of data
    for i in data:
        temp_feature = list(data[i]["data"]["original"].keys())[0]
        holdout_range = (0,int(len(data[i]["data"]["original"][temp_feature])*split[cur_spot])) if cur_spot == 0 else (int(len(data[i]["data"]["original"][temp_feature])*split[cur_spot-1]),int(len(data[i]["data"]["original"][temp_feature])*split[cur_spot]))
        total = 0
        features = list(data[i]["data"]["features"].keys())
        for j in range(len(data[i]["data"]["features"][features[0]])):

            # skip collection for items in holdout range
            if j >= holdout_range[0] and j < holdout_range[1]:
                continue

            temp = []
            for k in features:
                temp.append(data[i]["data"]["features"][k][j])

            if None not in temp:
                class_type = (data[i]["data"]["targets"].keys())[0]
                training_set[i]["base"][data[i]["data"]["targets"][class_type][j]].append(temp)
                total += 1
        for j in training_set[i]["base"]:
            training_set[i]["extracted"][j]["size"] = len(training_set[i]["base"][j])/total

    # print(training_set["Iris"]["extracted"]["Iris-setosa"]["size"]) # << This is debug

    # Then calculate the formula from #3 in the outline
    for i in data:
        for j in training_set[i]["base"]:
            features = list(training_set[i]["extracted"][j].keys())
            features.remove("size")
            for k in range(len(features)):
                # count value occurances
                for l in range(len(training_set[i]["base"][j])):

                    # skip entries in holdout range
                    if (l >= holdout_range[0] and l < holdout_range[1]):
                        continue

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

    # Now we can test the training set on the holdout values
    acc_entry = []
    for i in data:
        temp_acc = []
        temp_feature = list(data[i]["data"]["original"].keys())[0]
        size = len(data[i]["data"]["original"][temp_feature])
        temp_range = [0, int(split[cur_spot]*size)] if cur_spot == 0 else [int(split[cur_spot-1]*size), int(split[cur_spot]*size)]
        if temp_range[0] == temp_range[1]:
            temp_range[1] += 1
        features = list(data[i]["data"]["features"].keys())
        print(temp_range)
        for j in range(temp_range[1]-temp_range[0]):
            temp_record = []
            for k in data[i]["data"]["original"]:
                # print(temp_range[0] + j)
                temp_record.append(data[i]["data"]["original"][k][temp_range[0]+j])
            max_candidates = []
            for k in training_set[i]["base"]:
                likeliness = training_set[i]["extracted"][k]["size"]
                for l in range(len(features)):
                    try:
                        likeliness *= training_set[i]["extracted"][k][features[l]][temp_record[l]]
                    except:
                        likeliness *= training_set[i]["extracted"][k][features[l]]["default"]
                max_candidates.append(likeliness)
            max_ind = max_candidates.index(max(max_candidates))
            if list(training_set[i]["base"].keys())[max_ind] == temp_record[-1]:
                temp_acc.append(1)
            else:
                temp_acc.append(0)
        acc_entry.append(sum(temp_acc)/len(temp_acc))

    output = "[ "
    for i in acc_entry:
        output += "{:.2f}".format(i) + ", "
    output = output.rstrip(", ") + " ]"
    print(output)

    accuracies.append(acc_entry)

print("\nRaw Accuracies:")
for a in accuracies:
    output = "[ "
    for b in a:
        output += "{:.3f}".format(b) + ", "
    print(output.rstrip(", ") + " ]")

print("\nAverage Accuracies")
average = [0]*5
for a in accuracies:
    for b in range(len(a)):
        average[b] += a[b]
for a in range(len(average)):
    average[a] /= 10
output = "[ "
for a in average:
    output += "{:.3f}".format(a) + ", "
print(output.rstrip(", ") + " ]")

print("\nStatistical Significance (p-score against the null)")
print("\tNull Hypothesis:\n\tThe model's 10-fold test results are no\n\tmore (if not less) accurate than a blind guess\n\t(given that a blind guess will be correct \n\t1/total_classes of the time).")
# null hypothesis means
null_hyp = []
for i in data:
    null_hyp.append(1/len(list(training_set[i]["base"].keys())))
# standard deviation
sd = [0]*5
for a in accuracies:
    for b in range(len(a)):
        sd[b] += (a[b]-average[b])**2
for s in range(len(sd)):
    sd[s] /= 9
    sd[s] = sd[s]**0.5
# print(sd)
# statistic score (t-score)
t = []
for i in range(len(average)):
    t.append((average[i]-null_hyp[i])/(sd[i]/(10**0.5) if sd[i] != 0 else 0.0001))
# print(t)
p = []
for i in t:
    p.append(scipy.stats.t.sf(abs(i), 9))

for i in range(len(average)):
    # forcing hypothesis by only encouraging values outside of it
    if average[i] < null_hyp[i]:
        p[i] = 1
    
print("p-scores:")
output = "[ "
for i in p:
    output += str(i) + ", "
print(output.rstrip(", ") + " ]")

print("significant?:")
output = "[ "
for i in p:
    output += "insignificant" if i > 0.1 else "barely significant" if i > 0.5 else "somewhat significant" if i > 0.1 else "sigificant"
    output += ", "
print(output.rstrip(", ") + " ]w")
