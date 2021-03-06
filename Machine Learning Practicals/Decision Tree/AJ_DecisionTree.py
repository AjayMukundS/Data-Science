import numpy as np
#import graphviz
from leaf import leaf
import scipy.ndimage.filters as fltr

'''
Function: axis_partitions
Inputs: axis_data (This is the data pertaining to a single input feature)
Returns: partitions (This contains all the possible partitions that can be made for the given input feature)

This function finds all the possible partitions that can be made on an input feature based on the data.
It uses a uniform filter that computes the average of two consecutive values in a vector.
This is in fact the way by which the possible partitions are computed in the decision tree algorithm.
Remember, the values in the vector must be unique and sorted in ascending order.
'''
def axis_partitions(axis_data):
    partitions = fltr.uniform_filter(axis_data.astype(float), size = 2)
    partitions = np.delete(partitions, 0)

    return partitions

'''
Function: find_partitions
Inputs: data, in_feat (data = The dataset given, in_feat = The column names of the input features)
Returns: partitions (This contains all possible partitions that can be made for all input features)

This function sweeps through all the input features to find all the possible partitions to be evaluated in the tree formation.
It makes extensive use of the axis_partitions function.
Remember to send the vector of unique and sorted values from each feature to the axis_partitions function.
'''
def find_partitions(data, in_feat):
    partitions = {}
    
    #''' WRITE YOUR CODE HERE. '''

    return partitions

'''
Function: gini_index
Inputs: data, out_feat, classes (data = The dataset given, out_feat = The output feature column name, classes = The list of possible
                                 classes in the output feature)
Returns: E (The impurity)

This function computes the gini index for given data partition.
'''
def gini_index(data, out_feat, classes):

    #''' WRITE YOUR CODE HERE. '''

    return E

'''
Function: partition_evals
Inputs: data, partitions, out_feat, classes (data = The dataset given, partitions = The list of possible partitions from all input features,
                                             out_feat = The output feature column name, classes = The list of possible classes in the output
                                             feature)
Returns: max_feat (This is the input feature and its corresponding partition that obtained the maximum information gain)

This function makes use of the gini index to evaluate each partition and reports the partition that achieved the maximum information gain.
'''
def partition_evals(data, partitions, out_feat, classes):
    possible_partitions = {}

    #''' WRITE YOUR CODE HERE. '''

    max_feat = get_max_feat(possible_partitions)

    return max_feat

'''
Function: get_max_feat
Inputs: partition_data (This contains the information gain values for all the partitions)
Returns: max_feat (This is the input feature and its corresponding partition that obtained the maximum information gain)

This function reports the feature partitioning with highest information gain.
'''
def get_max_feat(partition_data):

    keys = list(partition_data.keys())
    max_feat = {}
    information_gain= []
    for k in keys:
    	information_gain.append(partition_data[k][0])
    max_val= 0
    for i in range(len(information_gain)):
        if information_gain[i]> max_val:
            max_val= information_gain[i]
            max_idx= i
    partition_list= list(partition_data.items())
    max_feat= {'max_feat': partition_list[max_idx][0], 'part_idx': partition_list[max_idx][1][1]}
    return max_feat

'''
Function: data_split
Inputs: data_parts, max_feats, k (data_parts = A list of data parts split from the original dataset based on the maximum information gain
                                  partitions, max_feats = A list of feature partitions that achieve max. information gain at all level of the
                                  decision tree formation so far, k = The index of the data part that has to be split)
Returns: None

This function splits the data into parts based on the partition that achieve max information gain, for further splitting at each level.
Even for the root node, the data_parts variable is encoded as {'0': data} with data = the entire dataset given.
The data_parts is recursively updated to store all the partitions that have been made.
'''
def data_split(data, column, value):
    index= data.columns
    for i in range(len(index)):
        if index[i]== column:
            idx= i
    dataset= data.iloc[:, :].values
    left, right = list(), list()
    for row in dataset:
        if row[idx] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

'''
Function: class_probs
Inputs: data_part, out_feat, classes (data_part = The specific data part given to evaluate the most probable output class, out_feat = The output
                                      feature column name, classes = The list of possible classes in the output feature)
Returns: p (The class probabilities)
'''
def class_probs(data_part, out_feat, classes):

    #''' WRITE YOUR CODE HERE. '''

    return p

'''
Function: add_stump
Inputs: tree, max_feat, pL, pR, classes, parent_node, branch (tree = The decision tree formed so far, max_feat = The newfeature partition that achieved
                                                              the max. information gain, pL = The class probabilities on the left side of the partition,
                                                              pR = The class probabilities on the left side of the partition, parent_node = The node at
                                                              which the new stump has to be added, branch = The branch of the parent node at the stump
                                                              must be added)
Returns: p_class (This tells if the classes predicted by the stump are final or there exist further partitions possible)

This function adds a new stump to the existing decision tree.
The root node is also added to an empty decision tree using this function.
'''
def add_stump(tree, max_feat, pL, pR, classes, parent_node = (), branch = 'T'):

    if np.max(pL) == 1.0:
        pL_class = 'fclass'
    else:
        pL_class = 'pclass'

    if np.max(pR) == 1.0:
        pR_class = 'fclass'
    else:
        pR_class = 'pclass'

    if parent_node != ():
        if branch == 'T':
            tree[parent_node]['T'] = max_feat
        else:
            tree[parent_node]['F'] = max_feat

    tree[max_feat] = {'T': (pL_class, classes[np.argmax(pL)]), 'F': (pR_class, classes[np.argmax(pR)])}
    p_class = [pL_class, pR_class]

    return p_class

'''
Function: prune
Inputs: tree, max_feats (tree = decision tree, max_feats = The list of all feature partitions made in the tree)
Returns: None

This function prunes the tree to have all stumps with unique class outputs (or leafs) at each branch.
'''
def prune(tree, max_feats):
    keys = list(max_feats.keys())

    inv_max_feats = {}
    for k in keys:
        inv_max_feats[max_feats[k]] = k

    keys = list(tree.keys())
    for k in keys:
        if tree[k]['T'][0][1:] == 'class' and tree[k]['F'][0][1:] == 'class':
            if tree[k]['T'][1] == tree[k]['F'][1]:
                cat = tree[k]['T']
                addr = inv_max_feats[k]
                idx = len(addr)-1
                
                parent_addr = addr[0: idx]

                parent = max_feats[parent_addr]
                if addr[idx] == '0':
                    tree[parent]['T'] = cat
                else:
                    tree[parent]['F'] = cat

                tree.pop(k)
    
'''
Function: build_tree
Inputs: data, in_feats, out_feat, depth (data = The entire given dataset, in_feats = The input feature column names, out_feat = The output feature column
                                         name, depth = The no. of levels the decision tree must have)
Returns: tree

This function build the decision tree
'''
def build_tree(data, in_feats, out_feat, depth = 1):
    tree = {}
    max_feats = {}
    
    categories = list(data[out_feat].unique())
    data_parts = {'0': data}
    p_class = {'0': 'pclass'}

    for d in range(depth):
        keys = list(data_parts.keys())

        for k in keys:
            if p_class[k] == 'pclass':

                #''' WRITE YOUR CODE HERE THAT FINDS ALL POSSIBLE PARTITIONS AND GETS THE BEST PARTITION FOR SPLITTING. '''
                #'''(Make use of the find_partitions and partitions_evals functions)'''
    
                max_feats[k] = max_feat
                data_split(data_parts, max_feats, k)
    
                pL = class_probs(data_parts[k + '0'], out_feat, categories)
                pR = class_probs(data_parts[k + '1'], out_feat, categories)

                if k == '0':
                    parent = ()
                    branch = 'T'
                else:
                    idx = len(k)-1
                    parent = max_feats[k[0: idx]]

                    if k[idx] == '0':
                        branch = 'T'
                    else:
                        branch = 'F'

                p_classes = add_stump(tree, max_feat, pL, pR, categories, parent, branch)
                p_class[k + '0'] = p_classes[0]
                p_class[k + '1'] = p_classes[1]

    prune(tree, max_feats)

    return tree

'''
Function: to_plot_tree
Inputs: tree, leaf_classes (tree = The tree formed by build_tree, leaf_classes = The output classes as leaf objects)
Returns: tree_plot (This is a tree made for plotting purposes)

This function converts the tree formed by build_tree into a plottable tree for visualization purposes.
'''
def to_plot_tree(tree, leaf_classes):
    tree_plot = {}

    key_list = list(tree.keys())

    for stump in key_list:
        if tree[stump]['T'][0] == 'pclass' or tree[stump]['T'][0] == 'fclass':
            T_node = leaf_classes[tree[stump]['T'][1]]
        else:
            T_node = tree[stump]['T'][0] + ' < ' + str(tree[stump]['T'][1])

        if tree[stump]['F'][0] == 'pclass' or tree[stump]['F'][0] == 'fclass':
            F_node = leaf_classes[tree[stump]['F'][1]]
        else:
            F_node = tree[stump]['F'][0] + ' < ' + str(tree[stump]['F'][1])
        
        tree_plot[stump[0] + ' < ' + str(stump[1])] = {'T': T_node, 'F': F_node}

    return tree_plot

'''
Function: plot
Inputs: tree, filename (tree = The plottable tree, filename = The file name without extension on which the decision tree is to be plotted)
Return: None

This function plots the plottable tree to an image in a .pdf file specified by the filename.
'''
def plot(tree, filename):
    dot_file = open(filename + '.dot', 'w')
    dot_file.write('digraph Tree {\n')
    dot_file.write('size = "4,4";\n\n')

    c = 0
    for start, d in tree.items():
        for weight, end in d.items():
            if isinstance(end, leaf):
                end_str = 'class_' + str(c)
                class_str = 'class_' + str(c) + ' [shape = box, color = gray, style = filled, label = "' + end.label + '"];\n'
                c += 1
            else:
                end_str = '"' + end + '"'
                class_str = ''
            
            dot_file.write('"' + start + '"' + ' -> ' + end_str + ' [label = "' + weight + '"];\n')
            dot_file.write(class_str)

    dot_file.write('}')
    dot_file.close()

    dot_data = open(filename + '.dot').read()

    #graph = graphviz.Source(dot_data)
    #graph.render(filename)
