'''

# Running
$ python -m streamlit run bhammin1_PA4.py

Brandi Hamming
Programming 4
Creates a KNN visualizer
User Can do the following
Select the query point. Data to make prediction on
Normalize the Data
Display two features in a scatter plot
'''

import streamlit as st
import random
from typing import List, Dict, Tuple, Callable
import math
from copy import deepcopy
from pprint import pprint
import matplotlib.pyplot as plt


def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data
	
def create_folds(xs: List, n: int) -> List[List[List]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test

def euclidean_dist(data:List[List[float]],query:List[float])->List[float]:
    '''
	This function calculates the euclidean distance between a group of training data
	compared to a single point. 
	'''
    dist = []
    for rw in data:
        if len(rw) != len(query):
            print("The number of features between data and query doesn't match")
            return []
        sum = 0
        for col_index in range(len(rw) -1): #excluding making distance on label field
            diff = rw[col_index] - query[col_index]
            diff_squared = diff * diff
            sum += diff_squared
        diff_root = math.sqrt(sum)
        dist.append(diff_root)
    return dist

def find_top_k(dists: List[float], k:int)->List[int]:
	# Finds the indexes of the closests neighbors. The closest k indexes are retuned
    dists_copy = deepcopy(dists)
    closest_neighbors = []
    while len(closest_neighbors) < k:
        current_min = min(dists_copy)
        neighbor = dists.index(current_min)
        closest_neighbors.append(neighbor)
        
        dists_copy.remove(current_min) # remove from list to find next closest neighbor

    return closest_neighbors

def knn(data:List[List[float]], queries:List[List[float]], k:int)-> Tuple[List[float],List[List[int]]]:
    '''
	- Returns a list of predictions for each query and the K nearest neighbors
	- The prediction is from taking the average value of the
	k nearest neighbors. 
	- The neareness measure comes from the euclidean distance

    Small change for program 4: returns list of neighbor indexes
	'''
    preds = []
    preds_neighbors = []

    for query in queries:
        dists = euclidean_dist(data,query) # get distances
        
        neighbors = find_top_k(dists, k) # find closest neighbors
        
        label_index = len(query) - 1
        labels = [data[n_index][label_index] for n_index in neighbors] # getting values of neighbors
        
        est_y = round((sum(labels)/ k), 2) # make prediction for query point
        preds.append(est_y) # add column
        preds_neighbors.append(neighbors) # update 
		
    
    return preds, preds_neighbors

def mse(labels:List[float], preds:List[float])-> float:
    '''
	- Returns the mean squared error of the predictions
	- This function first finds the error between the prediction and actual value
	- Then squares that value. 
	- Then the average is take of the errors. 
	- A low error rate is best. The closer to zero the better. 
	'''
    sum = 0
    for index in range(len(labels)):
        diff = labels[index] - preds[index]
        diff_squared = diff * diff
        sum += diff_squared
    mse_val = round((sum / len(labels)),4)
    return mse_val

def z_score(data:List[List[float]], query:List[List[float]])-> List[List[float]]:
    '''
    - Normalizes the data but transforming each column to be the z score
    - All columns in the query data set are transformed except for the label column
    - The label column is always the last column
    - Z = X- Mean / Standard Deviation
    '''
    means = []
    st_devs = []
    norm = deepcopy(query)

    for feature in range(len(data[0]) -1): # get data mean and stdev
        col = [row[feature] for row in data]
        mean = sum(col)/ len(col)
        diff = [ pow(( x- mean), 2) for x in col ]
        diff_sum = sum(diff)
        st_dev = pow(diff_sum/ len(col),.5)
        
        means.append(mean)
        st_devs.append(st_dev)
  
    for rw in range(len(query)):
        for col in range(len(query[rw])-1):
            z = (query[rw][col] - means[col])/ (st_devs[col])
            norm[rw][col] = round(z, 4) # normalize data
    return norm



def get_col_info(data:List[List[float]])-> Tuple[List[float],List[float]]:
    '''
    Finds the min and max value for each column
    This is used for setting the range for the query point
    '''
    mins = []
    maxes = []
  
    for i in range(len(data[0])):
        col = [row[i] for row in data] # get all records for a given column
        min_value = min(col)
        max_value = max(col)
        
        mins.append(min_value)
        maxes.append(max_value) 

    return  mins, maxes

def format_neighbors(data:List[List[float]],neighbors:List[List[int]], col_names:List[str])->dict:
    '''
    Returns the neighbors in a dictionary format. Used to display a table
    '''
    neighbors_df = {}
    neighbors_cols = [data[n_index] for n_index in neighbors[0]] # all indexes of neighbors
    # for this problem  neighbors is a list size of 1 with indexes because only one query point
    
    for i in range(len(neighbors_cols[0])):
        col = [row[i] for row in neighbors_cols] # get all rows for a given column
        neighbors_df[col_names[i]] = col
    return neighbors_df

    
def get_query(col_names:List[str], col_types:List[str], col_mins:List[int], col_maxes:List[int])-> Tuple:
    '''
    Function creates the side panel
    User can select the following
    Number of K
    Each query column value
    Normalize the data or not
    Two Features to display in a graph

    Function returns query dictionary, list of containing the query point
    ,k value, the normalization indicator, list of feature names to display
    '''
    query_dict = {}
    query_list = []
    with st.sidebar:
        st.markdown('## Select Number of Neighbors')
        # select parameter
        k = st.number_input('k Neighbors',min_value = 1, max_value = 10)  # this is a widget

        st.markdown('## Personalize Your Query Point ')

        # add each query column to side bar
        for i in range(len(col_names)): 
            col_label = col_names[i] +"\t" + col_types[i]
            col_value = st.number_input(col_label, min_value = col_mins[i], max_value = col_maxes[i], key = i)
            query_dict[col_names[i]] = [col_value]
            query_list.append(deepcopy(col_value))
        
        norm_ind = st.checkbox('Normalize the Data') # add normalization check box
        vis_fields = st.multiselect('Fields to Visualize',col_names,max_selections=2)
    return query_dict, [query_list], k, norm_ind, vis_fields

def get_query_norm_dict(query_list:List[List[float]], col_names:List[str])-> dict:
    '''
    Returns the values of the normalized query points into a dictionary
    Dictionary format needed to display a table
    '''
    query_dict = {}
    for i in range(len(col_names)):
        query_dict[col_names[i]] =[ query_list[0][i]] # query list is always length of 1
    return query_dict

def get_xy_names(features:List[str], col_names:List[str]) -> Tuple[List[str],List[int] ]:
    '''
    In case the user does not select 2 features
    Ensures two features are selected. Returns a list of col names and indexes
    that will be used as the x and y when plotting
    '''
    features_copy = deepcopy(features)
    feature_index = []
    
    # handle case of user not selecting col features
    if len(features) == 0: # no cols selected
        features_copy = [col_names[0], col_names[1]]
    if len(features) == 1:# only one col selected
        not_used_features = list(set(col_names)-set(features))
        features_copy.append(not_used_features[0]) # compare against one feature not already selected
    
    # get indexes
    feature_index.append(col_names.index(features_copy[0]))
    feature_index.append(col_names.index(features_copy[1]))
    
    return features_copy,feature_index

def get_xv_values(data:List[List[float]], features:List[int])-> Tuple[List[float], List[float]]:
    '''
    Returns x and y values for given data set
    Used for plotting
    '''
    x = [row[features[0]] for row in data]
    y = [row[features[1]] for row in data]

    return x, y


def plot_features(features:Tuple[List[str],List[int]], data:List[List[float]], neighbors:List[List[float]], query:List[List[float]]):
    '''
    plot_features
    Displays the two features in a scatter plot
    The query point, neighbors and data displayed separately 
    ''' 
    neighbors_data = []
    data_copy = deepcopy(data)
    for n in neighbors[0]:
        neighbors_data.append(data[n])
        data_copy.pop(n)
    # get x, y
    data_x, data_y = get_xv_values(data_copy, features[1])
    neightbors_x, neightbors_y = get_xv_values(neighbors_data, features[1])
    query_x, query_y = get_xv_values(query, features[1])
    # plot
    data_plot  = plt.scatter( data_x, data_y, color='y')
    query_plot = plt.scatter(query_x, query_y,  color='b')
    neighbor_plot = plt.scatter(neightbors_x, neightbors_y, marker='x',color='c')
    plt.legend((query_plot,neighbor_plot, data_plot),('Query', 'Neighbor', 'Data'),scatterpoints=1)
    plt.title("2 Feature Comparison")
    plt.xlabel(features[0][0])
    plt.ylabel(features[0][1])
    st.pyplot(plt.show())

def main():
    col_names = ["cement"
	, "slag"
	,"ash"
	,"water"
	,"superplasticizer"
	,"coarse aggregate"
	,"fine aggregate"
	,"age"
	,"concrete compressive strength"# y value
	]
    col_types = ["kg","kg","kg","kg","kg","kg","kg","days", "MPa"]

    st.set_page_config(layout="wide") # set page to fill the screen more
    st.set_option('deprecation.showPyplotGlobalUse', False) # remove plot warning
    
    data = parse_data("concrete_compressive_strength.csv")
    folds = create_folds(data, 10)
    mins, maxes = get_col_info(data)
    train, test = create_train_test(folds, 0)

    query_dict, query_list, k, norm, vis_fields = get_query(col_names, col_types, mins, maxes)

    # if data is normalized, then update train, query data
    if norm:
        query_list = z_score(train, query_list)
        query_dict = get_query_norm_dict(query_list, col_names)
        train = z_score(train, train)
    
    preds , neighbors = knn(train, query_list, k)
    mse_val = mse([query_list[0][-1]], preds)
    n_df = format_neighbors(train,neighbors, col_names)

    # format results on UI
    st.markdown('# K Nearest Neighbor Explorer')
    st.markdown('### Query Record')
    st.table(query_dict) # actual table

    st.markdown('**Prediction**: '+str(preds[0])+' '+ col_types[-8])

    # show the stake holder is the error rate is bad or not with an emoji
    if mse_val < 100:
        st.markdown('**Error Rate**: '+str(mse_val)+' Mean Squared Error '+':smile:')
    else:
        st.markdown('**Error Rate**: '+str(mse_val)+' Mean Squared Error '+':disappointed:')

    st.markdown('### Neighbors')
    st.table(n_df) # actual table

    # plot the features
    xy = get_xy_names(vis_fields, col_names)
    plot_features(xy, train, neighbors, query_list)


main()


