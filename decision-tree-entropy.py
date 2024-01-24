import math
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.children = {}
#calculate entropy
def entropy(groups, classes):
    total_samples = sum(len(group) for group in groups)
    entropy_val = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p > 0:
                score -= p * (math.log2(p))
        entropy_val += (size / total_samples) * score
    return entropy_val


def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def create_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#splitter to find the best split point
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            entropy_val = entropy(groups, class_values)
            if entropy_val < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy_val, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


#create decision tree
"""
this function create decision tree by recursively splitting 
the dataset and creating nodes until certain stopping conditions are reach,
such as reaching the maximum depth or having groups 
smaller than the specified minimum size
"""
def build_tree(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = create_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = create_terminal(left), create_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = create_terminal(left)
    else:
        node['left'] = get_split(left)
        build_tree(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = create_terminal(right)
    else:
        node['right'] = get_split(right)
        build_tree(node['right'], max_depth, min_size, depth + 1)

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def make_predictions(tree, dataset):
    predictions = [predict(tree, row) for row in dataset]
    return predictions

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('{}[X{} < {}]'.format(depth*' ', node['index'], node['value']))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('{}[{}]'.format(depth*' ', node))



train_data = pd.read_excel('trainDATA.xlsx')
train_dataset = train_data.values.tolist()

test_data = pd.read_excel('testDATA.xlsx')
test_dataset = test_data.values.tolist()

#i chose depth 5
max_depth = 5
min_size = 1

root = get_split(train_dataset)
build_tree(root, max_depth, min_size, 1)


print_tree(root)


real_acceptability = [row[-1] for row in test_dataset]
predictions = make_predictions(root, test_dataset)

#calculate print accuracy
correct_predictions = sum(1 for pred, actual in zip(predictions, real_acceptability) if pred == actual)
accuracy = correct_predictions / len(test_dataset) * 100
print("Accuracy: {:.2f}%".format(accuracy))

#my predictions to Excel file
result_df = pd.DataFrame({'Predicted_Acceptability': predictions, 'Actual_Acceptability': real_acceptability})
result_df.to_excel('predictions.xlsx', index=False)



