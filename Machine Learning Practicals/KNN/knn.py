import numpy as np
import copy
import pandas as pd
from math import sqrt
import cross_validation as cv

#Normalizing the data
def normalize(data):
	normalized_data= (data.values - np.mean(data.values))/np.std(data.values)
	out= {'mean':mean, 'std': std, 'norm_data': normalized_data}
	return out

#Finding the mean of the given dataset
def mean(data):
	mean= np.mean(data)
	return(mean)

#Finding the Standard Deviation of the given dataset
def std(data):
	std= np.std(data)
	return(std)

#Finding the euclidean distance
def distance(CV_dict, location, fold= 0):
	df_temp = copy.deepcopy(CV_dict[fold]['train'])
	idx= CV_dict[fold]['validation'].index.values
	df_temp['distance'] = np.sum((CV_dict[fold]['train'].values - CV_dict[fold]['validation'].loc[idx[location]].values)**2, axis = 1)
	df_temp= df_temp.sort_values(by = 'distance')
	idx = df_temp.sort_values(by= 'distance').index.values
	out= {'data': df_temp, 'index': idx}
	return(out)

#Finding the mean of the K nearest neighbours
def findMean(idx, data, K=3):
	K_idx = idx[0: K]
	mean = 0
	sum= 0
	for i in K_idx:
		sum += data.loc[i]['Spending Score (1-100)']
	mean= sum/ K
	out= {'mean': mean, 'K_idx': K_idx}
	return(out)
	
#Estimating the error for a particular value of K
def ErrorForK(data, CV_dict, k_fold, K= 5):
	Total_error= 0
	for fold in range(k_fold):
	    sum= 0
	    for location in range(CV_dict[fold]['validation'].shape[0]):
	        out= distance(CV_dict,location, fold)
	        idx= out['index']
	        df_temp= out['data']
	        out= findMean(idx, data, K)
	        mean= out['mean']
	        K_idx= out['K_idx']
        	Prediction_error= cv.Prediction_error(data, mean, location)
        	#print(Prediction_error)
	        sum= sum+ (Prediction_error* Prediction_error)
	    ErrorOfValidationSet= sum/ CV_dict[fold]['validation'].shape[0]
	    Total_error= Total_error+ ErrorOfValidationSet
	AverageErrorPerFold= Total_error/k_fold
	return(AverageErrorPerFold)


