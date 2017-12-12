
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np


# In[80]:


from random import seed
from random import randrange
from math import sqrt
from csv import reader


# In[81]:


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    #print('in cross validation algo')
    #print(dataset)
    dataset_split = []
    dataset_copy = list(dataset)
    #print('i am the copy')
    #print(dataset_copy)
    fold_size = int(len(dataset) / n_folds)
    #print(len(dataset_copy))
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            #print('index')
            #print(index)
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    #print('out cross validation algo')
    return dataset_split


# In[82]:


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    #print('in accuracy metric')
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    #print('out accuracy metric')
    return correct / float(len(actual)) * 100.0


# In[83]:


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    #print('in evaluate algo')
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        #train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    #print('out evaluate algo')
    return scores


# In[84]:


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    #print('in test split')
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    #print('out test split')
    return left, right


# In[85]:


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    #print('in gini index')
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
            # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    #print('out gini index')
    return gini


# In[86]:


# Select the best split point for a dataset
def get_split(dataset, n_features):
    #print('in get split')
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    #print(' n_features')
    #print(n_features)
    #print(len(features))
    #print('going in while')
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
            break
    #print('out while len')
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    #print('out get split')
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# In[87]:


# Create a terminal node value
def to_terminal(group):
    #print('in to terminal')
    outcomes = [row[-1] for row in group]
    #print('out to terminal')
    return max(set(outcomes), key=outcomes.count)


# In[88]:


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    #print('in split')
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
    #print('out split')


# In[89]:


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    #print('in build tree')
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    #print('out build tree')
    return root


# In[90]:


# Make a prediction with a decision tree
def predict(node, row):
    #print('in predict')
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


# In[91]:


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    #print('in subsample')
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    #print('out subsample')
    return sample


# In[92]:


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    #print('in bagging predict')
    predictions = [predict(tree, row) for tree in trees]
    #print('out bagging predict')
    return max(set(predictions), key=predictions.count)


# In[93]:


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    #print('in random forest')
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    #print('out random forest')
    return(predictions)


# In[142]:


# Test the random forest algorithm
seed(2)
seed(2)
# load and prepare data
#filename = 'input/test3.csv'
data_type={'is_booking':bool,'srch_ci' : np.str_, 'srch_co' : np.str_,
           'srch_adults_cnt' : np.int32, 'srch_children_cnt' : np.int32,
           'srch_rm_cnt' : np.int32, 'srch_destination_id':np.int32,
           'user_location_country' : np.int32, 'user_location_region' : np.int32,
           'user_location_city' : np.int32, 'hotel_cluster' : np.int32,
           'orig_destination_distance':np.float64, 'date_time':np.str_,
           'hotel_market':np.int32}

data_type1={'is_booking':bool, 'cnt':np.int32, 'hotel_cluster' : np.int32,'srch_destination_id':np.int32}
           # 'srch_adults_cnt' : np.int32, 'srch_children_cnt' : np.int32,'user_location_country' : np.int32, 'user_location_region' : np.int32,
          # 'user_location_city' : np.int32}

data_frame = pd.read_csv('train.csv',dtype=data_type1, usecols=data_type, parse_dates=['date_time'] ,sep=',', nrows=25000).dropna()
data_frame = data_frame[(data_frame['hotel_cluster'] == 91)  | 
        (data_frame['hotel_cluster'] == 41) | 
        (data_frame['hotel_cluster'] == 65) | 
        (data_frame['hotel_cluster'] == 48) | 
        (data_frame['hotel_cluster'] == 25)]#| 
       # (data_frame['hotel_cluster'] == 33)| 
       # (data_frame['hotel_cluster'] == 98)| 
        #(data_frame['hotel_cluster'] == 18)| 
       # (data_frame['hotel_cluster'] == 21)]
data_frame['year'] = data_frame['date_time'].dt.year
# look for booked hotels only
dataset1 = data_frame.query('is_booking==True')
del data_frame
#X = dataset1.drop(['is_booking','year'], axis=1)
#del dataset1
dataset = dataset1.as_matrix()
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 10
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
n_trees = 1
scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
#print('Trees: %d' % n_trees)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

