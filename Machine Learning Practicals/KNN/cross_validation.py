import knn
import numpy as np
import copy
import pandas as pd

#Function to split the data according to the number of folds
def split_data(data, folds= 10):
	block_size= int(data.shape[0]/ folds)
	block_dict= {}
	for i in range(folds):
    		if i< folds- 1:
        		block_dict[i]= data.loc[i* block_size: (i+1)* block_size- 1]
    		else:
        		block_dict[i]= data.loc[i* block_size: ]
	return(block_dict)

#Partitioning the data into Validation and Training set
def ValidationSplit(data, folds= 10):
	CV_dict= {}
	data_dict= {}
	for i in range(folds):
    		data_dict['validation']= data[i]    
    		not_i= np.array(range(folds))
    		not_i= np.delete(not_i, i)
    		frames= []
    		for j in not_i:
        		frames.append(data[j])
    		data_dict['train']= pd.concat(frames)
    		CV_dict[i]= copy.deepcopy(data_dict)
	return(CV_dict)

#Calculating the Prediction error
def Prediction_error(data, mean, location= 0):
	model_prediction = mean
	actual_prediction = data.loc[location]['Spending Score (1-100)']
	pred_error = actual_prediction - model_prediction
	return(pred_error)

