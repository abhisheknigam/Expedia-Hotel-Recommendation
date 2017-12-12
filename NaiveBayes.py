import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import time

% matplotlib inline

millis1 = int(round(time.time() * 1000))

dtype={'is_booking':bool,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.int32,
        'user_location_country' : np.float64,
        'user_location_region' : np.float64,
        'user_location_city' : np.float64,
        'hotel_cluster' : np.float64,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.int32}

df0 = pd.read_csv('train.csv',dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',',nrows=10000000).dropna()

df2 = df0[(df0['hotel_cluster'] == 91)  | (df0['hotel_cluster'] == 41)]
#| (df0['hotel_cluster'] == 65) | (df0['hotel_cluster'] == 48) | (df0['hotel_cluster'] == 25)

df1 = df2[[
    'date_time',
    'user_location_country',
    'orig_destination_distance',
    'srch_ci',
    'srch_co',
    'srch_adults_cnt',
    'srch_children_cnt',
    'srch_rm_cnt',
    'srch_destination_id',
    'is_booking',
    'hotel_market',
    'hotel_cluster',
    ]]

df1['srch_ci']=pd.to_datetime(df1['srch_ci'],infer_datetime_format = True,errors='coerce')
df1['srch_co']=pd.to_datetime(df1['srch_co'],infer_datetime_format = True,errors='coerce')
df1['date_time']=pd.to_datetime(df1['date_time'],infer_datetime_format = True,errors='coerce')

df1['month']= df1['date_time'].dt.month
df1['plan_time'] = ((df1['srch_ci']-df1['date_time'])/np.timedelta64(1,'D')).astype(float)
df1['length_of_stay']=((df1['srch_co']-df1['srch_ci'])/np.timedelta64(1,'D')).astype(float)

n=df1.orig_destination_distance.mean()
df1['orig_destination_distance']=df1.orig_destination_distance.fillna(n)
df1.fillna(-1,inplace=True)

lst_drop=['date_time','srch_ci','srch_co','is_booking','hotel_market','user_location_country','srch_destination_id', 'year']

df1['year']=df1['date_time'].dt.year
test = df1[(df1['is_booking']== True) & (df1['year']==2014)]
train = df1[((df1['is_booking']== False) & (df1['year']==2014)) | (df1['year'] != 2014)]



test.drop(lst_drop,axis=1,inplace=True)
train.drop(lst_drop,axis=1,inplace=True)

expected_result = test.hotel_cluster

del test['hotel_cluster']

del df1
del df2

trainingSet = train.as_matrix()
testSet = test.as_matrix()

def segregateClass(df):
    classLabel = {}
    for i in range(len(df)):
        array = df[i]
        if (array[-4] not in classLabel):
            classLabel[array[-4]] = []
        classLabel[array[-4]].append(array)
    return classLabel

def calculateStandardDeviation(arrayOfNum):
    avg = calculateMean(arrayOfNum)
    variance = sum([pow(x-avg,2) for x in arrayOfNum])/float(len(arrayOfNum)-1)
    return math.sqrt(variance)

def calculateMean(arrayOfNum):
    return sum(arrayOfNum)/float(len(arrayOfNum))

def segregateClasses(dataset):
    classSummary = [(calculateMean(attribute), calculateStandardDeviation(attribute)) for attribute in zip(*dataset)]
    del classSummary[-4]
    return classSummary

def summarizeByClass(dataset):
    differentClasses = segregateClass(dataset)
    classSummaries = {}
    for value, instances in differentClasses.items():
        classSummaries[value] = segregateClasses(instances)
    return classSummaries

import math
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def probabilityForEachClass(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += calculateProbability(x, mean, stdev)
    return probabilities

def getClassPrediction(summaries, inputVector):
    probabilities = probabilityForEachClass(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def predict(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = getClassPrediction(summaries, testSet[i])
        predictions.append(result)
    return predictions

def accuracy(actual_values, predicted_values):
    match = 0
    for x in range(len(actual_values)):
        if actual_values['hotel_cluster'][x] == predicted_values[x]:
            match += 1
    return (match/float(len(actual_values))) * 100.0

summaries = summarizeByClass(trainingSet)

predicted_value = predict(summaries, testSet)

pd_final = pd.DataFrame(data=expected_result, dtype=np.int64)

pd_final = pd_final.reset_index()

accuracy = accuracy(pd_final, predicted_value)

accuracy

millis2 = int(round(time.time() * 1000))

final_time = (millis2 - millis1)/1000
final_time

#91,41,65,48,25,33,95,18,21,70
y = label_binarize(pd_final['hotel_cluster'], classes=[91,41])
predicted_value = label_binarize(predicted_value, classes=[91,41])

average_precision = average_precision_score(y, predicted_value)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

