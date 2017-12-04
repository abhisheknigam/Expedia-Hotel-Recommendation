from random import seed
from random import randrange
from csv import reader

import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import ml_metrics as metrics


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
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
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal tree_node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a tree_node or make terminal
def split(tree_node, max_depth, min_size, depth):
	left, right = tree_node['groups']
	del(tree_node['groups'])
	# check for a no split
	if not left or not right:
		tree_node['left'] = tree_node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		tree_node['left'], tree_node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		tree_node['left'] = to_terminal(left)
	else:
		tree_node['left'] = get_split(left)
		split(tree_node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		tree_node['right'] = to_terminal(right)
	else:
		tree_node['right'] = get_split(right)
		split(tree_node['right'], max_depth, min_size, depth+1)

# search in tree to find class
def search_tree(tree_node, row):
	if row[tree_node['index']] < tree_node['value']:
		if isinstance(tree_node['left'], dict):
			return search_tree(tree_node['left'], row)
		else:
			return tree_node['left']
	else:
		if isinstance(tree_node['right'], dict):
			return search_tree(tree_node['right'], row)
		else:
			return tree_node['right']

def decision_tree_algo(train, test, max_depth, min_size):
	
	root = get_split(train)
	
	split(root, max_depth, min_size, 1)
	
	results = list()
	
	for row in test: 
		result = search_tree(root, row)
		results.append(result)
	return(results)

# Evaluate decision tree algo using a 80/20 split
def process(dataset, no_of_folds, *args):
	
	dataset_split = list()

	dataset_copy = list(dataset)

	fold_size = int(len(dataset) / no_of_folds)

	for i in range(no_of_folds):
		
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)

	accuracies = list()

	for one_fold in dataset_split:

		training_set = list(dataset_split)

		training_set.remove(one_fold)

		training_set = sum(training_set, [])
		
		testing_set = list()
		
		for rec in one_fold:
			row = list(rec)
			testing_set.append(row)
			row[-1] = None
		
		print len(training_set)
		print len(testing_set)
		
		predicted = decision_tree_algo(training_set, testing_set, *args)

		actual = [row[-1] for row in one_fold]
		
		positive = 0
		
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				positive += 1 

		accuracy = (positive / float(len(actual)) * 100.0)

		accuracies.append(accuracy)

		# using 1 fold only here.. algo takes too much time for multiple folds
		# and different trees generally give close accuracy results when experimented with
		break
	
	return accuracies

seed(1)
data_type={'is_booking':bool,'srch_ci' : np.str_, 'srch_co' : np.str_,
           'srch_adults_cnt' : np.int32, 'srch_children_cnt' : np.int32,
           'srch_rm_cnt' : np.int32, 'srch_destination_id':np.int32,
           'user_location_country' : np.int32, 'user_location_region' : np.int32,
           'user_location_city' : np.int32, 'hotel_cluster' : np.int32,
           'orig_destination_distance':np.float64, 'date_time':np.str_,
           'hotel_market':np.int32}

data_frame = pd.read_csv('train_medium.csv',dtype=data_type, usecols=data_type, parse_dates=['date_time'] ,sep=',')
data_frame = data_frame[(data_frame['hotel_cluster'] == 91)  | (data_frame['hotel_cluster'] == 41) | (data_frame['hotel_cluster'] == 65) | (data_frame['hotel_cluster'] == 48) | (data_frame['hotel_cluster'] == 25)]#| (data_frame['hotel_cluster'] == 33)| (data_frame['hotel_cluster'] == 95)| (data_frame['hotel_cluster'] == 18)| (data_frame['hotel_cluster'] == 21)]
data_frame['year'] = data_frame['date_time'].dt.year

# look for booked hotels only
train_frame = data_frame.query('is_booking==True')

del data_frame

train_frame['srch_ci'] = pd.to_datetime(train_frame['srch_ci'], infer_datetime_format = True, errors='coerce')
train_frame['srch_co'] = pd.to_datetime(train_frame['srch_co'], infer_datetime_format = True, errors='coerce')

#derived attributes
train_frame['month']= train_frame['date_time'].dt.month
# calculate plan time from dates
train_frame['plan_time'] = ((train_frame['srch_ci']-train_frame['date_time'])/np.timedelta64(1,'D')).astype(float)
#calculate no of hotel nights from dates
train_frame['hotel_nights']=((train_frame['srch_co']-train_frame['srch_ci'])/np.timedelta64(1,'D')).astype(float)


col_drop_list = ['date_time','srch_ci','srch_co','user_location_country','user_location_region',
                 'srch_adults_cnt','srch_children_cnt', 'orig_destination_distance']
train_frame.drop(col_drop_list, axis=1, inplace=True)

X = train_frame.drop(['is_booking','year'], axis=1)

del train_frame

dataset = X.as_matrix()

print len(dataset)

no_of_folds = 5

max_depth = 3

min_size = 10

accuracies = process(dataset, no_of_folds, max_depth, min_size)
#print('accuracies: %s' % accuracies)
print('Mean Accuracy: %.3f%%' % (sum(accuracies)/float(len(accuracies))))