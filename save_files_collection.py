import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

import itertools

import os

import math

import pickle

#import keras specific functions for storing and loading models
from keras.models import Model

from keras.models import model_from_json





# save models to JSON -> check that weights_file name uses ".h5" format & model_file_name ".json"
def save_models_to_json(model_file_name, model_weights_file_name, Store_PATH, model):
    
    '''
    function stores keras model on disk --> models are stored in json-format
    --> weights and architecture of model are separately stored!
    '''
    
    #create paths for model architecture & weights:
    model_final_path = Store_PATH + model_file_name    
    weights_final_path = Store_PATH + model_weights_file_name

    #store model & weights:
    model_as_json = model.to_json()
    with open(model_final_path, "w") as json_file:
        json_file.write(model_as_json)
    # serialize weights to HDF5
    model.save_weights(weights_final_path)
    print("Saved model to disk")

    
    
def store_retrained_predictions(predictions_dict, model_dict, df_save_PATH, verbose=0):
    
    '''
    stores prediction results on disk which are kept in dict. --> only stores the most recent predictions for each date 
    
    #predictions are stored in one df: However only the most recent prediction for each timestamp 
    are stored (outdated predictions are discarded)

    '''
           
    key_list = list(predictions_dict.keys())
    #sort keylist by dates (ascending: 2011, 2012..):
    key_list.sort()
    
    model_name_list = list(model_dict.keys())

    for i in range(len(key_list)):

        if i < 1:
            all_preds_df = predictions_dict[key_list[i]][0]

        else:  
            #concat all preds underneath each other --> this way there should be a lot of duplicates!
            all_preds_df = pd.concat([all_preds_df,predictions_dict[key_list[i]][0]],axis=0)
            #check for duplicates:
            #delete first occurence of duplicates: "~" = negation of boolean --> this way the most recent prediction is kept
            all_preds_df = all_preds_df.loc[~all_preds_df.index.duplicated(keep='last')]
        
      
        if verbose == 1:
            print('preds of iteration : ', i+1)
            print('head after iteration: ', all_preds_df.head())
            print('tail after iteration: ', all_preds_df.tail())
            
    #store df on disk:
    results_df_filename = 'results_retrained_{}'.format(model_name_list[0])  + '.csv' #use model_name for file_name -> this way retraining schema is kept in name
    results_df_final_path = os.path.join(df_save_PATH, results_df_filename)

    all_preds_df.to_csv(results_df_final_path, header=True)

    print('predictions stored on disk!')
    
    
    return all_preds_df
    
        

        

def store_retrained_predictions_multiple_areas(predictions_dict, models_dict, df_Store_PATH):
    
    #store predictions on disk:

    key_list = list(predictions_dict.keys())
    
    #access dict to get model_name 
    model_name = list(models_dict['area237'].keys())
    #delete area label:
    model_name = model_name[0].replace('_area237','')
    
    All_areas_results_dict = {}
    
    for u in range(len(key_list)):
          
        #get predictions for each area:
        key_list_2 = list(predictions_dict[key_list[u]].keys())
        
        area_results_dict = predictions_dict[key_list[u]]
        
        for i in range(len(key_list_2)):

            #concat prediction results:
            #initialize df with first results of first model (should be very first year for which predictions are available)
            if i < 1:
                all_results_df = area_results_dict[key_list_2[i]][0]

            #concat remaining prediction results for each model underneath each other:
            else:
                results_df_year_i = area_results_dict[key_list_2[i]][0]

                all_results_df = pd.concat([all_results_df,results_df_year_i],axis=0)


        #sort entires in df (in case keys are not sorted):
        all_results_df.sort_index(inplace=True, ascending=True)
        
        #append result of single area:
        All_areas_results_dict[key_list[u]] = all_results_df
        
        
        
    #concat final predictions for all areas:  
    for w in range(len(key_list)):
           
            #initialize df with first results of first model (should be very first year for which predictions are available)
            if w < 1:
                final_results_df = All_areas_results_dict[key_list[w]]
                
                
            #concat remaining prediction results for each model next to each other:
            else:
                results_df_area_i = All_areas_results_dict[key_list[w]]

                final_results_df = pd.concat([final_results_df,results_df_area_i],axis=1)
  
    
    print('final shape: ', final_results_df.shape)       
                
    #store df on disk:
    results_df_filename = 'results_retrained_{}'.format(model_name)  + '.csv' #use model_name for file_name -> this way retraining schema is kept in name
    results_df_final_path = os.path.join(df_save_PATH, results_df_filename)

    final_results_df.to_csv(results_df_final_path, header=True)

    print('predictions stored on disk!')
    
    return final_results_df


    
    

def store_retrained_drift_detection_results(predictions_dict, model_dict, df_save_PATH, verbose=0):
    
    '''
    stores prediction results stored in dict on disk, by only storing the most recent predictions for each date 
    
    #predictions are stored in one df: However only the most recent prediction for each timestamp 
    are stored (outdated predictions are discarded)

    '''
           
    key_list = list(predictions_dict.keys())
    #sort keylist by dates (ascending: 2011, 2012..):
    key_list.sort()
    
    model_name_list = list(model_dict.keys())

    for i in range(len(key_list)):

        if i < 1:
            all_preds_df = predictions_dict[key_list[i]][0]

        else:  
            #concat all preds underneath each other --> this way there should be a lot of duplicates!
            all_preds_df = pd.concat([all_preds_df,predictions_dict[key_list[i]][0]],axis=0)
            #check for duplicates:
            #delete first occurence of duplicates: "~" = negation of boolean --> this way the most recent prediction is kept
            all_preds_df = all_preds_df.loc[~all_preds_df.index.duplicated(keep='last')]
        
      
        if verbose == 1:
            print('preds of iteration : ', i+1)
            print('head after iteration: ', all_preds_df.head())
            print('tail after iteration: ', all_preds_df.tail())
            
    #store df on disk:
    results_df_filename = 'results_retrained_{}'.format(model_name_list[0])  + '.csv' #use model_name for file_name -> this way retraining schema is kept in name
    results_df_final_path = os.path.join(df_save_PATH, results_df_filename)

    all_preds_df.to_csv(results_df_final_path, header=True)

    print('predictions stored on disk!')
    
    
    return all_preds_df
    
    
    
    
def store_model_and_history_on_disk(models_dict, model_save_PATH, df_save_PATH, verbose=0):


    #store models & history on disk:

    for key in models_dict:
        
        if verbose > 0:
            print('### Model to be stored: ', key)

        # 1) save models:
        model_to_store = models_dict[key][1]
        model_file_name = key + '.json'
        model_weights_file_name = key + '_weights.h5'


        #call function to store model:
        save_models_to_json(model_file_name, model_weights_file_name, model_save_PATH, model_to_store)


        # 2) save training history:
        #create df for traning_history:           
        training_history = models_dict[key][0]  #access history
        
        try:
            hist_col_labels = ['loss (mse)','mae','val_loss (mse)','val_mae']

            hist_df = pd.DataFrame(training_history.history['loss'],columns=[hist_col_labels[0]]) #create df of history
            hist_df[hist_col_labels[1]] = training_history.history['mean_absolute_error']
            hist_df[hist_col_labels[2]] = training_history.history['val_loss']
            hist_df[hist_col_labels[3]] = training_history.history['val_mean_absolute_error']
        
        except:
            
            hist_col_labels = ['loss (mse)','mae']

            hist_df = pd.DataFrame(training_history.history['loss'],columns=[hist_col_labels[0]]) #create df of history
            hist_df[hist_col_labels[1]] = training_history.history['mean_absolute_error']

        #store training_history on disk:
        hist_df_filename = 'history_' + key + '.csv'

        hist_df_final_path = os.path.join(df_save_PATH, hist_df_filename)
        hist_df.to_csv(hist_df_final_path, header=True)

        print('Save history_df on disk done')


        
        
def store_xg_boost_model_and_traininghistory(model_and_history_dict, error_type = 'rmse', model_save_PATH = '', 
                                             df_save_path = '', verbose=0):
    
    #store models & history on disk:

    for key in model_and_history_dict:
        
        if verbose > 0:
            print('### Model to be stored: ', key)
            
        #NOTE: "key" contains name of model 
        
        #1)
        #store model:
        model_to_store = model_and_history_dict[key][1]
            
        final_model_name = key + '.pickle.dat'

        file_to_save = model_save_PATH + final_model_name

        #save model on disk:
        pickle.dump(model_to_store, open(file_to_save,"wb"))
    
           
        #2)
        #get training history of training data:
        training_data_hist = model_and_history_dict[key][0]['validation_0'][error_type]
        valid_data_hist = model_and_history_dict[key][0]['validation_1'][error_type]


        all_hist_df = pd.DataFrame()
        all_hist_df['Training_Error'] = training_data_hist
        all_hist_df['Val_Error'] = valid_data_hist

        #store df on disk:
        hist_name = 'history_{}'.format(key)  + '.csv' #use model_name for file_name -> this way retraining schema is kept in name
        hist_final_path = os.path.join(df_save_path, hist_name)

        all_hist_df.to_csv(hist_final_path, header=True)

        print('History & Model stored on disk')
        
    
    return all_hist_df
        
             
        
        
        
def store_detected_change_dates(detected_dates_dict, df_store_PATH):
    '''
    function stores dict of dates at which change was detected as csv
    --> for each key: dates should be stored as a list of timestamps
    --> this function does not store extra columns to distinguish between retraining dates & weight updating dates!
    '''
    key_list = list(detected_dates_dict.keys())
    
    
    #get max length of detected dates:
    length_list = [len(detected_dates_dict[key]) for key in key_list]
    max_length = max(length_list)
    #print('max length: ', max_length)
    
    #initialize empty df to easily append new columns:
    all_dates_df = pd.DataFrame()
    
    length_list = [len(detected_dates_dict[key]) for key in key_list]
    
    for i in range(len(key_list)):
        #create list with dates as strings of each area:
        dates_list_area_i = [str(date) for date in detected_dates_dict[key_list[i]]]
        
        #check if a list exists:           
        delta_len = max_length - len(dates_list_area_i)
        
        if delta_len == max_length:
            dates_list_area_i = list(np.zeros(max_length))
        if delta_len > 0 and delta_len != max_length:
            help_li = list(np.zeros(delta_len))
            dates_list_area_i = dates_list_area_i + help_li
        
        #create column with list:
        all_dates_df[key_list[i]] = dates_list_area_i

    #store df on disk:
    file_name = 'detected_change_dates_per_area.csv'
    df_final_path = os.path.join(df_store_PATH, file_name)
    
    all_dates_df.to_csv(df_final_path, header=True)
    
    return all_dates_df


    
    
def store_detected_change_dates_with_index(detected_dates_dict, retrainining_dates_list, model_and_history_dict,
                                           switch_updating_dates_list=[], df_store_PATH=''):
    '''
    function stores dict of dates at which change was detected as csv
    --> for each key: dates should be stored as a list of timestamps
    -> this function stores extra columns to distinguish between retraining dates & weight updating dates!
    
    Params:
        detected_dates_dict = dict with all detected dates and keys = area labels
        retrainining_dates_list = list of dates at which complete model was retrained
        switch_updating_dates_list = if Switching Scheme was applied, list of dates at which model weights were updated
        
    '''
    
    model_names = list(model_and_history_dict.keys())
    
    key_list = list(detected_dates_dict.keys())
       
    #initialize empty df to easily append new columns:
    all_dates_df = pd.DataFrame()
       
    
    overall_dates_list = []
    area_labels_list = []
    
    for i in range(len(key_list)):
        #create list with dates as strings of each area:
        dates_list_area_i = [date for date in detected_dates_dict[key_list[i]]]
        
        overall_dates_list = overall_dates_list + dates_list_area_i
        
        help_area_list = []
        for u in range(len(dates_list_area_i)):
            
            help_area_list.append(key_list[i])
        
        
        area_labels_list = area_labels_list + help_area_list
        
    
    
    #2.1)
    #store results in df:
    all_dates_df['dates'] = overall_dates_list
    all_dates_df['area'] = area_labels_list
    
    #2.2)
    #set dates as index:
    all_dates_df = all_dates_df.set_index('dates')    
    #sort index:
    all_dates_df = all_dates_df.sort_index()
    
    #2.3)
    #add dates of retraining & switch weight updating if available:
    #create help_array filled with zeros to match length of "all_dates_df"
    help_li_retraining = np.zeros(len(all_dates_df['area'])-len(retrainining_dates_list))
    #append dates on top:
    help_li_retraining = np.append(retrainining_dates_list, help_li_retraining)
    
    #2.4)
    #create new column in df:
    all_dates_df['retraining_date'] = help_li_retraining
    
    #2.5)
    #check if list is available
    if switch_updating_dates_list:
        #create help_array filled with zeros to match length of "all_dates_df"
        help_li_weight_updates = np.zeros(len(all_dates_df['area'])-len(switch_updating_dates_list))
        #append dates on top:
        help_li_weight_updates = np.append(switch_updating_dates_list, help_li_weight_updates)
    
        #create new column in df:
        all_dates_df['switch_weight_update_date'] = help_li_weight_updates
    
    
    #2.7)
    #store df on disk:
    file_name = 'all_detected_change_dates_and_area_{}.csv'.format(model_names[0])
    df_final_path = os.path.join(df_store_PATH, file_name)    
    all_dates_df.to_csv(df_final_path, header=True)
    
    
    return all_dates_df





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    