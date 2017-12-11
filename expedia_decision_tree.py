from random import seed
from random import randrange
from csv import reader
import datetime
import pandas as pd
import numpy as np
import ml_metrics as metrics


def split_rows(i, val, dataset):
	first, second = list(), list()
	for rec in dataset:
		if rec[i] < val:
			first.append(rec)
		else:
			second.append(rec)
	return first, second

# Calculate GINI for split
def gini_index(splits, classes):
	n = float(sum([len(split) for split in splits]))
	gini_value = 0.0
	for split in splits:
		size = float(len(split))
		if size == 0:
			continue
		result = 0.0
		for class_val in classes:
			prob = [row[-1] for row in split].count(class_val) / size
			result += prob * prob
		gini_value += (1.0 - result) * (size / n)
	return gini_value

# Select split attribute
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_splits = 9999, 9999, 9999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			splits = split_rows(index, row[index], dataset)
			gini = gini_index(splits, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_splits = index, row[index], gini, splits
	return {'index':b_index, 'value':b_value, 'splits':b_splits}

# create terminal node woth class
def create_leaf_node(split):
	vals = [row[-1] for row in split]
	return max(set(vals), key=vals.count)

# create tree
def create_tree(tree_node, max_depth, min_size, depth):
	left, right = tree_node['splits']
	del(tree_node['splits'])
	if not left or not right:
		tree_node['left'] = tree_node['right'] = create_leaf_node(left + right)
		return
	if depth >= max_depth:
		tree_node['left'], tree_node['right'] = create_leaf_node(left), create_leaf_node(right)
		return
	if len(left) <= min_size:
		tree_node['left'] = create_leaf_node(left)
	else:
		tree_node['left'] = get_split(left)
		create_tree(tree_node['left'], max_depth, min_size, depth + 1)
	if len(right) <= min_size:
		tree_node['right'] = create_leaf_node(right)
	else:
		tree_node['right'] = get_split(right)
		create_tree(tree_node['right'], max_depth, min_size, depth + 1)

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
	
	create_tree(root, max_depth, min_size, 1)
	
	results = list()
	
	for rec in test: 
		result = search_tree(root, rec)
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

data_frame = pd.read_csv('train.csv',dtype=data_type, usecols=data_type, parse_dates=['date_time'] ,sep=',')
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
print('Mean Accuracy: %.3f%%' % (sum(accuracies)/float(len(accuracies))))