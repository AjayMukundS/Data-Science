import pandas as pd

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini
    
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
    
# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            #print( ' X%d < %.3f Gini=%.3f ' % ((index+1), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return { ' index ' :b_index, ' value ' :b_value, ' groups ' :b_groups}
    
# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
    
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node[ ' groups ' ]
    del(node[ ' groups ' ])
    # check for a no split
    if not left or not right:
        node[ ' left ' ] = node[ ' right ' ] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node[ ' left ' ], node[ ' right ' ] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node[ ' left ' ] = to_terminal(left)
    else:
        node[ ' left ' ] = get_split(left)
        split(node[ ' left ' ], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node[ ' right ' ] = to_terminal(right)
    else:
        node[ ' right ' ] = get_split(right)
        split(node[ ' right ' ], max_depth, min_size, depth+1)
        
# Make a prediction with a decision tree
def predict(node, row):
    if row[node[ ' index ' ]] < node[ ' value ' ]:
        if isinstance(node[ ' left ' ], dict):
            return predict(node[ ' left ' ], row)
        else:
            return node[ ' left ' ]
    else:
        if isinstance(node[ ' right ' ], dict):
            return predict(node[ ' right ' ], row)
        else:
            return node[ ' right ' ]
            
# Build a decision tree
def build_tree(dataset, max_depth, min_size):
    root = get_split(dataset)
    #print(root)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print( ' %s[Var0%d < %s] ' % ((depth* ' ' , (node[ ' index ' ]+1), node[ ' value ' ])))
        print_tree(node[ ' left ' ], depth+1)
        print_tree(node[ ' right ' ], depth+1)
    else:
        print( ' %s[%s] ' % ((depth* ' ' , node)))


