import pandas as pd
import numpy as np


import datetime
from dateutil.relativedelta import relativedelta

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

import itertools

import os

import math

import gc

import scipy.stats as st


import drift_detection as ddt
    
###################################### Functions for Actual Retraining ###########################################


    
def multiple_differencing(dataset,shift_interval_list):
    '''
    applies differencing multiple times based on given list
    '''
    dataset_copy = dataset.copy(deep=True)
    
    for i in range(len(shift_interval_list)):
        shifted_df = dataset_copy.shift(shift_interval_list[i])
        diff_series = dataset_copy - shifted_df
        diff_series.dropna(inplace=True)
        
        #reassign dataset_copy:
        dataset_copy = diff_series
        
    return diff_series
    
    
    

def get_current_predictions(predictions_dict, verbose=0):
       
    key_list = list(predictions_dict.keys())
    #sort keylist by dates (ascending: 2011, 2012..):
    key_list.sort()


    for i in range(len(key_list)):

        if i < 1:
            all_preds_df = predictions_dict[key_list[i]][0]

        else:  
            #concat all preds underneath each other --> this way there should be a duplicates if predictions overlap 
            all_preds_df = pd.concat([all_preds_df,predictions_dict[key_list[i]][0]],axis=0)
            #check for duplicates:
            #delete first occurence of duplicates: "~" = negation of boolean --> this way the most recent prediction is kept
            all_preds_df = all_preds_df.loc[~all_preds_df.index.duplicated(keep='last')]
        
      
        if verbose == 1:
            print('preds of iteration : ', i+1)
            print('head after iteration: ', all_preds_df.head())
            print('tail after iteration: ', all_preds_df.tail())
    
    return all_preds_df
    

    
    
    


def start_retraining(year_list, model_name, model_instance, org_ts_series, XGBoost_flag = False,
                     window_step_size=1, update_weights_flag=False, index_of_start_train_year=0, forecast_range_days = 365,
                     index_of_last_train_year=1, n_epochs_weight=150, n_epochs_retrain = 150, overwrite_params = False,
                     ceiling_flag_valid = False, ceiling_flag_test = False, 
                     end_of_dataset = '2018-06-30 22:00:00', modelcounter = 0, get_preds_flag = True, verbose=0):     
    
    '''
    
    #function retrains given model and returns new model + predcitions based on given years for training & predictions
    Note: n_epochs can be adjusted if retraining should be done with less epochs
    
    most important: year_list:
    
    year_list= years to train model & make predictions 
    
    Example: 
        year_list = ['2009','2010','2011','2012'] --> '2009','2010' used for training, predictions made for '2011' & '2012'
        year_list = ['2009-02','2010-04','2011-03','2012-04'] --> '2009-02'-'2010-04' used for training, predictions made for '2011-03'-2012-03' & '2012-04'-'2013-04'

    '''
    
    
    '''
    Note: it is assumed that years selected for training data are among the first years (indices) of year_list 
            --> indices [0,1..]
            
    Parameters:
    
    n_epochs_weight: number of epochs to update weights of model
    
    n_epochs_retrain: number of epochs to train new model
    
    overwrite_params: Flag need to be set to "TRUE" if n_epochs_retrain should be different than n_epochs stored in model_instance 
    
    get_preds_flag:
        Explanation of "get_preds_flag":
            if set to FALSE, no predictions are made directly with the newly trained model based on given valid and test set

            instead of setting "get_preds_flag" to FALSE, one can also set the dates for valid & test set to None

    '''
    
    
    #nested function:
    def year_slicing_assignment(date_valid, date_test, ceiling_flag_valid, ceiling_flag_test, end_of_dataset, 
                                forecast_range_days, verbose=0):

        ''' nested function checks if exactly one year is selected or specific Timestamp: 
            (Note: timestamp has max length of 19) >> Then function assigns years correctly which are used to slice the data_df


            if not a single year but also month or even hour is given, we need to define "start_" & "end_" date for slicing 

            Note: validation_set & test_set are always set to lenght given by "forecast_range_days"

        '''

        if verbose > 0:
            print('>> Dates are assigned...')
            print('>date_valid: ', date_valid)
            print('>date_test: ', date_test)
            
        #check for invalid input:

        if date_valid == None and date_test == None:
            print('No predictions are made, model is only retrained or weights are updated')
            assignment = ()
            return assignment


        # 1) set valid year:
        #1.1) set valid year to regular timestamp:
        if len(str(date_valid)) > 4 and ceiling_flag_valid == False:

            #set valid year:
            helper_stamp_valid = pd.Timestamp(str(date_valid)) #necessary to reconvert a timestamp!
            start_validation_set_year = str(helper_stamp_valid)
            end_validation_set_year = str(helper_stamp_valid + relativedelta(days=forecast_range_days)) #set length of one year
            
            if verbose > 0:
                print('case 1.1) start_validation_set_year : ', start_validation_set_year)
                print('case 1.1) end_validation_set_year : ', end_validation_set_year)
            
        # 1.2) #set valid year up to last entry in dataset:
        if len(str(date_valid)) > 4 and ceiling_flag_valid == True:

            #ceil valid year to only make predictions up to end of dataset:
            #set valid year:
            helper_stamp_test = pd.Timestamp(str(date_valid))
            start_validation_set_year = str(helper_stamp_test)
            end_validation_set_year = str(pd.Timestamp(end_of_dataset)) #directly set end of date_valid as end of dataset

            if verbose > 0:
                print('case 1.2) start_validation_set_year : ', start_validation_set_year)
                print('case 1.2) end_validation_set_year : ', end_validation_set_year)
         
        
            
        # 2) set test year:
        #2.1) set test year to regular timestamp:
        if date_test != None and len(str(date_test)) > 4 and ceiling_flag_test == False:

            #set test year:
            helper_stamp_test = pd.Timestamp(str(date_test))
            start_test_set_year = str(helper_stamp_test)
            end_test_set_year = str(helper_stamp_test + relativedelta(days=forecast_range_days))
            
            if verbose > 0:
                print('case 2.1) start_test_set_year : ', start_test_set_year)
                print('case 2.1) end_test_set_year : ', end_test_set_year)

        # 2.2) #set test year up to last entry in dataset:
        if date_test != None and len(str(date_test)) > 4 and ceiling_flag_test == True and ceiling_flag_valid == False:

            #set test year:
            helper_stamp_test = pd.Timestamp(str(date_test))
            start_test_set_year = str(helper_stamp_test)
            end_test_set_year = str(pd.Timestamp(end_of_dataset)) #directly set end of test_set_year as end of dataset

            if verbose > 0:
                print('case 2.2) start_test_set_year : ', start_test_set_year)
                print('case 2.2) end_test_set_year : ', end_test_set_year)
                

        # 2.3) #set test year to same timestamp as validation year
        if date_test == None or (ceiling_flag_test == True and ceiling_flag_valid == True):
            #if date_test is set to "None" it means end_validation_set_year already reached end of dataset 
            # --> just use values of validation set again
            start_test_set_year = start_validation_set_year
            end_test_set_year = start_validation_set_year
            
            if verbose > 0:
                print('case 2.3) start_test_set_year : ', start_test_set_year)
                print('case 2.3) end_test_set_year : ', end_test_set_year)

        # 3) set both test year & valid year if "whole" years are given (e.g. '2011', '2012'):
        if date_test != None and len(str(date_valid)) == 4 and len(str(date_test)) == 4:  
            #only a complete year is given --> set "end_sets" to None since they are not needed
            start_validation_set_year = str(date_valid)
            start_test_set_year = str(date_test) 
            end_validation_set_year = None
            end_test_set_year = None
            
            if verbose > 0:
                print('case 3) start_validation_set_year : ', start_validation_set_year)
                print('case 3) start_test_set_year : ', start_test_set_year)
                print('case 3) end_validation_set_year : ', end_validation_set_year)
                print('case 3) end_test_set_year : ', end_test_set_year)



        assignment = (start_validation_set_year, end_validation_set_year, start_test_set_year, end_test_set_year)

        return assignment
 
    
 
    print('selected years for training: ' , year_list[index_of_start_train_year:index_of_last_train_year+1])
    print('year_list given: ', year_list)

    
    
    # 1) assign years for slicing:
    #Note: years have to be converted to strings for slicing dfs.. 
    start_train_year = str(year_list[index_of_start_train_year])
    last_train_set_year = str(year_list[index_of_last_train_year])
    #create list to store years for which predictions should be made:
    val_test_years_list = year_list[(index_of_last_train_year+1):]
    #set last_date to last entry of list, since we need last_date to get last month of predictions
    if year_list[-1] == None:
        last_date = None
    else:
        last_date = str(year_list[-1])   
    
                 
    # 2) get size of training data:
    training_delta = (pd.Timestamp(last_train_set_year) - pd.Timestamp(start_train_year)) 
    training_size_in_days = training_delta.days
    training_info = '{}_s{}_{}_e{}_{}'.format(training_size_in_days, pd.Timestamp(start_train_year).month, 
                                              pd.Timestamp(start_train_year).year, 
                                              pd.Timestamp(last_train_set_year).month, 
                                              pd.Timestamp(last_train_set_year).year)
    
    # 3) get number of days to predict & last year of predictions:
    if last_date == None:
        
        last_year_preds = pd.Timestamp(end_of_dataset)
        last_year_preds = '{}_{}'.format(last_year_preds.month, last_year_preds.year)
            
    else:

        if len(last_date) > 4:
            last_year_preds = pd.Timestamp(last_date) + relativedelta(years=1)
            last_year_preds = '{}_{}'.format(last_year_preds.month, last_year_preds.year)

        else:
            last_year_preds = pd.Timestamp(last_date)
            last_year_preds = '{}_{}'.format(last_year_preds.month, last_year_preds.year)
        
    
    # 4) modify model name & add details of training:
    #model_name example:
    #model_name = 'multivar_lstm_2H_{}_{}_batch512_drop03_clip_norm_shuffle_scaling_tanh_W{}_20largest_areas__trainsize{}_stepsize{}_preds{}__y{}'.format(n_hidden_neurons_1,n_hidden_neurons_2, n_timesteps, training_size, window_step_size, number_of_pred_years, test_set_year) 

    model_name = model_name + '_count{}__trainsize{}__stepsize{}__p{}'.format(modelcounter,training_info, window_step_size, last_year_preds)                   


    # 5) get data for training & predictions:
    end_of_dataset_stamp = pd.Timestamp(end_of_dataset)
    dataset_last_year = end_of_dataset_stamp.year   
    ts_series = org_ts_series[start_train_year:str(dataset_last_year)]
    
    
    # 6) create dicts to store model & results; add model_name as key:
    model_dict = {}
    model_results_dict = {}  #stores prediction results of model 

    model_dict[model_name] = []
    
    #store overall avg. RMSE of valid & testsets in list:
    all_rmse_results = []
    

    # 7) start process to retrain model:

    print('#### Train model: {} ####'.format(model_name))

    # 7.1) initialize model with first year of list: 
    #set validation year & test year to same year! -> this way we can get predictions for different "validation_sets"
    
    #call function to get correct assignment of years:
    '''NOTE: very first time of retraining, year_slicing is called only with valid set..'''
    assign_results = year_slicing_assignment(val_test_years_list[0],val_test_years_list[0], ceiling_flag_valid, 
                                             ceiling_flag_test, end_of_dataset=end_of_dataset, 
                                             forecast_range_days=forecast_range_days, verbose = verbose)
       
    
    #if all dates are set to "None" "assign_results" is an empty tuple, --> adjust years: dates are set to an arbitrary value in dataset e.g. start dates, since preds are not stored
    if not assign_results:

        start_validation_set_year = pd.Timestamp(last_train_set_year)
        end_validation_set_year = start_validation_set_year
        start_test_set_year =  start_validation_set_year #test_set_year only needed as input for function; results are not stored yet! --> therefore set to same date as valid_year         
        end_test_set_year = start_validation_set_year
        
        skip_preds_flag = True
        
    else:
        skip_preds_flag = False
        
        start_validation_set_year = assign_results[0]
        end_validation_set_year = assign_results[1]
        start_test_set_year =  start_validation_set_year #test_set_year only needed as input for function; results are not stored yet! --> therefore set to same date as valid_year         
        end_test_set_year = start_validation_set_year
    
    
    if get_preds_flag == False:
        print('>> No preds are returned, only training history & model')
        #no preds are returned 
        skip_preds_flag = True
    
    
    if verbose ==1:
        
        print('>start_train_year: ', start_train_year)
        print('>last_train_set_year: ', last_train_set_year)
         
        print('start_validation_set_year: ', start_validation_set_year)
        print('end_validation_set_year: ', end_validation_set_year)
        print('start_test_set_year: ', start_test_set_year)
        print('end_test_set_year: ', end_test_set_year)
    

    # 7.2) call function to either update weights or create & train new model:
    
    #check if XGBoost model is used:
    
    if XGBoost_flag == False:
    
        if update_weights_flag == True:
            results_years_i = model_instance.update_model_weights(ts_series, start_train_year, last_train_set_year, 
                                                                  start_validation_set_year, start_test_set_year, 
                                                                  model_name, end_validation_set_year= end_validation_set_year, 
                                                                  end_test_set_year=end_test_set_year, n_epochs = n_epochs_weight,
                                                                  get_preds_flag = get_preds_flag, verbose=verbose)

        else:
            results_years_i = model_instance.retrain_model(ts_series, start_train_year, last_train_set_year, 
                                                           start_validation_set_year, start_test_set_year, 
                                                           model_name, end_validation_set_year=end_validation_set_year, 
                                                           n_epochs = n_epochs_retrain, overwrite_params = overwrite_params,
                                                           end_test_set_year=end_test_set_year, get_preds_flag = get_preds_flag,
                                                           verbose=verbose)
            
    else:
            
        if update_weights_flag == True:
            results_years_i = model_instance.update_model_weights(ts_series, start_train_year, last_train_set_year, 
                                                                  start_validation_set_year, start_test_set_year, 
                                                                  model_name, end_validation_set_year= end_validation_set_year, 
                                                                  end_test_set_year=end_test_set_year, n_estimators = n_epochs_weight,
                                                                  overwrite_params = overwrite_params,
                                                                  get_preds_flag = get_preds_flag, verbose=verbose)

        else:
            results_years_i = model_instance.retrain_model(ts_series, start_train_year, last_train_set_year, 
                                                           start_validation_set_year, start_test_set_year, 
                                                           model_name, end_validation_set_year=end_validation_set_year, 
                                                           n_estimators = n_epochs_retrain, overwrite_params = overwrite_params,
                                                           end_test_set_year=end_test_set_year, get_preds_flag = get_preds_flag,
                                                           verbose=verbose)
        
            

    #Note: training_data does not change within one retraining iteration! -> therefore we only store model & history for first training within this iteration; predictions for different years are made separately in the following loop (except first year prediction)
    
    # 7.3) only append model itself, modelname, history & org. ts_series:
    model_dict[model_name].append(results_years_i[0]) #append history
    model_dict[model_name].append(results_years_i[1]) #append model

    # 7.4) only append predictions of model for first year:
    #add years as key to store only most recent predictions:
    if skip_preds_flag == False:
        model_results_dict[start_validation_set_year] = []
        model_results_dict[start_validation_set_year].append(results_years_i[5]) #only append results of validation_year
    
        #store RMSE result:
        all_rmse_results.append(results_years_i[7][0])
         
    
    # 7.5) get predictions for remaining years:
    if skip_preds_flag == False: 
        for u in range(1,len(val_test_years_list),2):

            #select years:
            if u < len(val_test_years_list) and (u+1) < len(val_test_years_list):

                #set years:           
                #call function to get correct assignment of years:
                assign_results = year_slicing_assignment(val_test_years_list[u],val_test_years_list[u+1], ceiling_flag_valid,
                                                         ceiling_flag_test, end_of_dataset=end_of_dataset, 
                                                         forecast_range_days = forecast_range_days,
                                                         verbose = verbose)
                
                
                #check assignment of dates:
                if not assign_results:
                    #stop loop if all dates are set to "None"
                    print('No dates left --> break')
                    break
                
                else:
                    start_validation_set_year = assign_results[0]
                    end_validation_set_year = assign_results[1]
                    start_test_set_year =  assign_results[2] 
                    end_test_set_year = assign_results[3]

                if verbose ==1:
                    print('val_test_years_list[u]: ', val_test_years_list[u])
                    print('val_test_years_list[u+1]: ', val_test_years_list[u+1])
                    print('start_validation_set_year: ', start_validation_set_year)
                    print('end_validation_set_year: ', end_validation_set_year)
                    print('start_test_set_year: ', start_test_set_year)
                    print('end_test_set_year: ', end_test_set_year)

                #add years as key to store only most recent predictions:
                model_results_dict[start_validation_set_year] = []
                model_results_dict[start_test_set_year] = []

                #call function to get predictions:
                #Note: in model_instance the current prediction_model is already stored and gets called by the following function:
                validation_results, predictions_results, rmse_val, rmse_test =  model_instance.generate_data_get_predictions(ts_series, start_train_year, 
                                                                                                        last_train_set_year, start_validation_set_year, 
                                                                                                        start_test_set_year, 
                                                                                                        end_validation_set_year=end_validation_set_year, 
                                                                                                        end_test_set_year=end_test_set_year,
                                                                                                        verbose=0)

                # append validation & prediction results of model:
                model_results_dict[start_validation_set_year].append(validation_results)
                model_results_dict[start_test_set_year].append(predictions_results)

                #append RMSE results:
                all_rmse_results.append(rmse_val[0])
                all_rmse_results.append(rmse_test[0])


            #if only one year is left:
            elif u < len(val_test_years_list):

                #set years:
                #check if entry is None
                if val_test_years_list[u] == None:
                    #break loop since "None" indicates that years were already processed by loop above & predictions were already made for respective years
                    break


                #call function to get correct assignment of years:
                assign_results = year_slicing_assignment(val_test_years_list[u],val_test_years_list[u], ceiling_flag_valid, 
                                                         ceiling_flag_test, end_of_dataset=end_of_dataset, 
                                                        forecast_range_days = forecast_range_days,
                                                        verbose = verbose)

                start_validation_set_year = assign_results[0]
                end_validation_set_year = assign_results[1]
                start_test_set_year =  assign_results[2] 
                end_test_set_year = start_test_set_year #make test_set shorter since only validation results are stored


                if verbose ==1:
                    print('val_test_years_list[u]: ', val_test_years_list[u])
                    print('start_validation_set_year: ', start_validation_set_year)
                    print('end_validation_set_year: ', end_validation_set_year)
                    print('start_test_set_year: ', start_test_set_year)
                    print('end_test_set_year: ', end_test_set_year)


                #add years as key to store only most recent predictions:
                model_results_dict[start_validation_set_year] = []

                #call function to get predictions:
                #Note: in model_instance the current prediction_model is already stored and gets called by the following function:
                validation_results, predictions_results, rmse_val, rmse_test =  model_instance.generate_data_get_predictions(ts_series, start_train_year, 
                                                                                                        last_train_set_year, start_validation_set_year, 
                                                                                                        start_test_set_year, 
                                                                                                        end_validation_set_year=end_validation_set_year, 
                                                                                                        end_test_set_year=end_test_set_year,
                                                                                                        verbose=0)

                # only append validation results of model:
                model_results_dict[start_validation_set_year].append(validation_results)


                #append RMSE results:
                all_rmse_results.append(rmse_val[0])
    
    
    
    #call garbage collector:
    gc.collect()
    
    # model_dict = contains training history & model itself | model_results_dict contains predictions of newly created model for given dates | all_rmse_results contains RMSE results of preds
    return  model_dict, model_results_dict, all_rmse_results


    
      
    

def make_predictions_retrained_or_regular(year_list, model_instance, org_ts_series, model_name, forecast_range_days = 365,
                                          ceiling_flag_valid= False, ceiling_flag_test = False, 
                                          end_of_dataset = '2018-06-30 22:00:00', 
                                          verbose=0):     
    
    '''
    
    #function makes predcitions based on given years for training & predictions
    --> years can be either Timestamps like '2011-01-30 22:00:00' or whole years like '2012'
    
    >> "regular" = a single year is given, like '2012'
    >> "retrained" = a Timestamp is given
    
    most important: year_list:
    
    year_list= years to train model & make predictions 
    
    Example: 
        year_list = ['2009','2010','2011','2012'] --> '2009','2010' used for training, predictions made for '2011' & '2012'
        year_list = ['2009-02','2010-04','2011-03','2012-04'] --> '2009-02'-'2010-04' used for training, predictions made for '2011-03'-2012-03' & '2012-04'-'2013-04'
    
    
    
    Parameters:
        
        forecast_range_days = how many days should be predicted into the future
        
        ceiling_flag_valid = Flag that makes sure that no predictions are made for observations that are out of the dataset
        
        ceiling_flag_test = Flag that makes sure that no predictions are made for observations that are out of the dataset
        
        end_of_dataset = Last entry in dataset
        
    '''
    
    
    '''
    !!!! IMPORTANT !!!!
    Note: it is assumed that years selected for training data are among the first years (indices) of year_list 
            --> indices [0,1..]
    '''
    
    #nested function:
    def year_slicing_assignment(date_valid, date_test, ceiling_flag_valid, ceiling_flag_test, end_of_dataset, 
                                forecast_range_days, verbose=0):

        ''' nested function checks if exactly one year is selected or specific Timestamp: 
            (Note: timestamp has max length of 19) >> Then function assigns years correctly which are used to slice the data_df


            if not a single year but also month or even hour is given, we need to define "start_" & "end_" date for slicing 

            Note: validation_set & test_set are always set to length of one year!

        '''
        
        if verbose > 0:
            print('Dates are assigned...')
            print('date_valid: ', date_valid)
            print('date_test: ', date_test)
            
            
        #check for invalid input:
        if date_valid == None and date_test == None:
            print('>> No predictions are made !! no valid input!!')
            print('date_valid: ', date_valid)
            print('date_test: ', date_test)
            
            assignment = ()
            return assignment
        
        
        # 1) set valid year:
        #1.1) set valid year to regular timestamp:
        if len(str(date_valid)) > 4 and ceiling_flag_valid == False:

            #set valid year:
            helper_stamp_valid = pd.Timestamp(str(date_valid)) #necessary to reconvert a timestamp!
            start_validation_set_year = str(helper_stamp_valid)
            end_validation_set_year = str(helper_stamp_valid + relativedelta(days=forecast_range_days)) #set length of one year

            if verbose > 0:
                print('case 1.1) start_validation_set_year : ', start_validation_set_year)
                print('case 1.1) end_validation_set_year : ', end_validation_set_year)

        # 1.2) #set valid year up to last entry in dataset:
        if len(str(date_valid)) > 4 and ceiling_flag_valid == True:

            #ceil valid year to only make predictions up to end of dataset:
            #set valid year:
            helper_stamp_test = pd.Timestamp(str(date_valid))
            start_validation_set_year = str(helper_stamp_test)
            end_validation_set_year = str(pd.Timestamp(end_of_dataset)) #directly set end of date_valid as end of dataset

            if verbose > 0:
                print('case 1.2) start_validation_set_year : ', start_validation_set_year)
                print('case 1.2) end_validation_set_year : ', end_validation_set_year)
         
        
            
        # 2) set test year:
        #2.1) set test year to regular timestamp:
        if date_test != None and len(str(date_test)) > 4 and ceiling_flag_test == False:

            #set test year:
            helper_stamp_test = pd.Timestamp(str(date_test))
            start_test_set_year = str(helper_stamp_test)
            end_test_set_year = str(helper_stamp_test + relativedelta(days=forecast_range_days))

            if verbose > 0:
                print('case 2.1) start_test_set_year : ', start_test_set_year)
                print('case 2.1) end_test_set_year : ', end_test_set_year)

        # 2.2) #set test year up to last entry in dataset:
        if date_test != None and len(str(date_test)) > 4 and ceiling_flag_test == True and ceiling_flag_valid == False:

            #set test year:
            helper_stamp_test = pd.Timestamp(str(date_test))
            start_test_set_year = str(helper_stamp_test)
            end_test_set_year = str(pd.Timestamp(end_of_dataset)) #directly set end of test_set_year as end of dataset

            if verbose > 0:
                print('case 2.2) start_test_set_year : ', start_test_set_year)
                print('case 2.2) end_test_set_year : ', end_test_set_year)
                

        # 2.3) #set test year to same timestamp as validation year
        if date_test == None or (ceiling_flag_test == True and ceiling_flag_valid == True):
            #if date_test is set to "None" it means end_validation_set_year already reached end of dataset 
            # --> just use values of validation set again
            start_test_set_year = start_validation_set_year
            end_test_set_year = start_validation_set_year

            if verbose > 0:
                print('case 2.3) start_test_set_year : ', start_test_set_year)
                print('case 2.3) end_test_set_year : ', end_test_set_year)

        # 3) set both test year & valid year if "whole" years are given (e.g. '2011', '2012'):
        if date_test != None and len(str(date_valid)) == 4 and len(str(date_test)) == 4:  
            #only a complete year is given --> set "end_sets" to None since they are not needed
            start_validation_set_year = str(date_valid)
            start_test_set_year = str(date_test) 
            end_validation_set_year = None
            end_test_set_year = None
            
            if verbose > 0:
                print('case 3) start_validation_set_year : ', start_validation_set_year)
                print('case 3) start_test_set_year : ', start_test_set_year)
                print('case 3) end_validation_set_year : ', end_validation_set_year)
                print('case 3) end_test_set_year : ', end_test_set_year)



        assignment = (start_validation_set_year, end_validation_set_year, start_test_set_year, end_test_set_year)

        return assignment
    


    print('selected years for training: ' , year_list[0:2])
    print('year_list given: ', year_list)

    
    
    # 1) assign years for slicing:
    #Note: years have to be converted to strings for slicing dfs.. 
    start_train_year = str(year_list[0])
    last_train_set_year = str(year_list[1])
    #create list to store years for which predictions should be made:
    val_test_years_list = year_list[2:]                
        
    # 2) get data for training & predictions:
    end_of_dataset_stamp = pd.Timestamp(end_of_dataset)
    dataset_last_year = end_of_dataset_stamp.year
    ts_series = org_ts_series[start_train_year:str(dataset_last_year)]


    # 3) create dicts to store model & results; add model_name as key:
    model_results_dict = {}  #stores prediction results of model 
           
    #store overall avg. RMSE of valid & testsets in list:
    all_rmse_results = []

    # 4) get predictions:

    print('#### Make predictions model: {} ####'.format(model_name))

    # 4.1) initialize model with years of list: 
   
    #call function to get correct assignment of years/dates:
    assign_results = year_slicing_assignment(val_test_years_list[0],val_test_years_list[1], ceiling_flag_valid, 
                                             ceiling_flag_test, end_of_dataset=end_of_dataset, 
                                             forecast_range_days=forecast_range_days, verbose = verbose)
    
    
    #if all dates are set to "None" "assign_results" is an empty tuple, 
    if not assign_results:
        print('No valid input given --> break')
        #return empty dict and list:
        return model_results_dict, all_rmse_results
        
        
    else:   
        start_validation_set_year = assign_results[0]
        end_validation_set_year = assign_results[1]
        start_test_set_year =  assign_results[2] 
        end_test_set_year = assign_results[3]
    
    
    if verbose ==1:
        
        print('>start_train_year: ', start_train_year)
        print('>last_train_set_year: ', last_train_set_year)
        
        print('start_validation_set_year: ', start_validation_set_year)
        print('end_validation_set_year: ', end_validation_set_year)
        print('start_test_set_year: ', start_test_set_year)
        print('end_test_set_year: ', end_test_set_year)
    
    
    
    # 4.2) get predictions:
    #add years as key to store only most recent predictions:
    model_results_dict[start_validation_set_year] = []
    model_results_dict[start_test_set_year] = []
    
    #call function to get predictions:
    #Note: in model_instance the current prediction_model is already stored and gets called by the following function:
    validation_results, predictions_results, rmse_val, rmse_test =  model_instance.generate_data_get_predictions(ts_series, start_train_year, 
                                                                                            last_train_set_year, start_validation_set_year, 
                                                                                            start_test_set_year, 
                                                                                            end_validation_set_year=end_validation_set_year, 
                                                                                            end_test_set_year=end_test_set_year, 
                                                                                            verbose=verbose)
    # append validation & prediction results of model:
    
    #check if only preds for "valid set" are made or also for "test set":
    delta_start_dates = pd.Timestamp(start_validation_set_year) - pd.Timestamp(start_test_set_year)
    if delta_start_dates.days == 0:
        
        if verbose > 0:
            print('only preds of validset are stored..')
        
        #only append preds of "valid set", since test set is not needed (set to None):
        model_results_dict[start_validation_set_year].append(validation_results)
        
        #append RMSE:
        all_rmse_results.append(rmse_val[0])
        
        
    else:
        # append BOTH: validation & prediction results of model:
        model_results_dict[start_validation_set_year].append(validation_results)
        model_results_dict[start_test_set_year].append(predictions_results)
    
        #append RMSE:
        all_rmse_results.append(rmse_val[0])
        all_rmse_results.append(rmse_test[0])

    
    
    #call garbage collector:
    gc.collect()
    
    return model_results_dict, all_rmse_results
    
  
    
    
def drift_detection_retraining(model_instance, org_ts_series, model_name='model_name', detector_type ='PH', 
                               XGBoost_flag = False, converted_stream_flag=False, use_differenced_ts = False, 
                               differencing_scheme = [168,24],
                              n_epochs_retrain = 150, overwrite_params = False,
                              n_epochs_weight = 150, update_weights_flag = False, update_retrain_switch = False, 
                              weight_update_range = [3], weight_update_backshift = 1, 
                              make_preds_with_weight_range = False,
                              adjust_lags_flag = True,
                              detect_dates_range_days = 365,
                              start_train_year = '2009', last_train_set_year = '2010', 
                              first_date_dataset = '2009-01-01 00:00:00',
                              start_of_preds_date = '2011-01-01 00:00:00',
                              end_of_dataset_date = '2018-06-30 22:00:00', 
                              first_forecast_range_days = 365, 
                              forecast_window_days = 7,
                              window_step_size=1, index_of_start_train_year=0, index_of_last_train_year=1, 
                              sensitivity=1, sensitivity_type='monthly', 
                              verbosity=0):
    
    '''
    
    function applies drift detection on given series and retrains/updates model if drift was detected. 
    Function stops if all predictions are made for the given years
    Function retruns all predictions up to the last given year
    
    
    Note: "ts_20largest" is taken as default input for converting predictions --> make sure it is available!!
    
    
    Parameters:
    
        #set params to select years for retraining:
        window_step_size = 1 #number of years the window is shifted 

        index_of_start_train_year = 0 #set index where "start_train_year" is stored in list (if set to "0" always first entry within window is selected as start_train_year)
        index_of_last_train_year = 1 #set index where "last_train_year" is stored in list (if set to "1" always second entry within window is selected as last_train_year)


        #set training specific parameters:
        
        use_differenced_ts = whether to difference ts to apply drift detectors on rather stationary time series or not
        
        differencing_scheme = differencing which should be applied on time series
        
        n_epochs_weight = epochs to use for weight updating
        
        n_epochs_retrain = epochs to use for retraiing the model / creating new model
        overwrite_params = needed, if epochs should differ from defaul n_epochs stored in model_instance.n_epochs
        
        update_weights_flag = Flag to update weights or retrain new model
        
        
        update_retrain_switch = indicates a specific training scheme: model weights are updated if drift is detected within one year after last_train_set_year
                                                                      if drift is detected after this time interval, model is discarded and new model is trained!


        
        weight_update_range = range of years within weight updating in a "switching" scheme is accepted (equals "tau" in Thesis):
                             --> the input is used to calculate the difference of the first date of training_set 
                                 (the trainingset which was recently used to retrain the model) and the detected drift date
                                 --> if difference greater than "weight_update_range" years, a new model is trained instead of updating the weights of the existing model
                                
                            --> default = 3 
                                Example:
                                    if model was trained 2009:2010, first date of traning_set = 2009-01-01
                                    --> weights of model are updated if drifts belong to 2011
                                    if drift in 2012 --> model is retrained since 2012 is more than 3 years apart from 2009
                             
        

        weight_update_backshift = number of years which should be included to perform the weight updating of a model 
                                  Note: only affects the weight updating if "Switching scheme" is performed (equals "alpha" in Thesis)
                                
                                Example:
                                    if "weight_update_backshift" = 1:
                                        based on recent detected drift date, the observations up to 1 year before the drift are included in the training set
                                        --> drift at 2011-01-10 --> training set for weight updating: 2010-01-10:2011-01-10
                                    if "weight_update_backshift" = 0:
                                        only most recent "unseen" observations are included:
                                        --> drift detected at 2011-05-10 and previous drift at 2011-01-10:
                                        -> trainingset:  2011-01-10:2011-05-10
                                               
                                        
         make_preds_with_weight_range = if Switching Scheme is applied, Flag indicates that predictions with newly trained model should be scaled based on same training set that was used for the weight updating step
                                        --> if set to FALSE: to scale the inputs for the predictions, the training set is "increased" by considering all observations from very last complete retraining up to the current drift date
                                        --> if set to TRUE: to scale the inputs for the predictions, the training set is includes only data in time frame: first date (= "drift date" - weight_update_backshift) up to second date (= 'drift date') (which equals the years of "backshift")
                                             --> e.g. if set to "TRUE" and backshift = 2 --> 2 years of data which is used to update the model incrementally is also used to scale the next input data
       
        
        detect_dates_range_days = Difference in days accepted between two detected drifts to update the model weights instead of creating a new model
                                 --> only relevant if "Switching Scheme" is applied
        
        
        converted_stream_flag = "True": the datastream to monitor is based on converted predictons
                                "False:" the datastream to monitor is based on the actual observations
        
        
        detector_type = type of detector used for drift detection: 'ADWIN', 'PH', 'HDDDM', 'STEPD', 'MK', 'EDDM'
        
        
        sensitivity = how many drifts are accepted to occur until function should stop and detected dates should be returned --> Note: sensitivity_type is always monitored 
                    --> for instance, even if sensitivity is set to 3 but sensitivity_type condition is not met by second drift, the function breaks
                    --> if  sensitivity = 3, and all three drifts meet the sensitivity_type condition, function breaks automatically after 4th drift is detected
                    
        sensitivity_type = number of days two subsequent detected drifts are allowed to differ form each other 
                        "monthly" = 30 days
                        "quarterly" = 3*30 days
                        "yearly" = 365 days

                        if set to "monthly" --> if two detected drifts differ more than 30 days, function breaks and returns second last drift date
                        if set to "monthly" --> if two detected drifts differ less than 30 days --> function waits for another detected drift date and checks again the difference

        
        
    '''
    
    
    #nested function to check if assigned dates are correct or already out of range:
    def check_assigned_dates(assigned_dates_list, end_of_dataset, forecast_range_days, verbose=0):
                
        '''
        #nested function to check if assigned dates are correct or already out of range:
        --> if out of range or almost end of dataset, adjust dates...
        
        Parameters:
        
        
        Returns:
        
        ceiling_flag_valid = if set to TRUE, indicates that end of dataset is almost reached and preds should 
                            only be made up to last date in dataset
        ceiling_flag_test = if set to TRUE, indicates that end of dataset is almost reached and preds should 
                            only be made up to last date in dataset
                            --> Note: only relevant if a test_set is used for preds along with the valid set
        
        stop_preds_flag = indicates that all preds are made and no further can be made
        
        
        
        Note: Function assumes that test_set makes preds based on valid_set:
            for instance:
                if valid_set makes preds for next 6 days, test_set is assumed to also make preds for the next 6 days but starts 
                at last date of preds from valid_set + 1hr 
                --> valid_preds: 01-01 10:00:00 to 01-06 10:00:00
                --> test_preds: 01-06 11:00:00 to 01-11 11:00:00
        
        '''
        
        print('## Assigned Dates are double checked..')
        
        #assign dates to check each date separately:
        start_date_valid_set = assigned_dates_list[2]
        start_date_test_set = assigned_dates_list[3]
                
        
        # 2) check for exceptions in dates and set flags to handle exceptions and/or adjust dates:
        
        #set flags to default params: ---> flags are adjusted in the next conditions if needed
        ceiling_flag_valid = False
        ceiling_flag_test = False
        stop_preds_flag = False
        
        #2.1) check if valid_set or test_set is out of range:
        check_date_forecast = pd.Timestamp(end_of_dataset) - relativedelta(days=forecast_range_days) 
        check_date_double_forecast = pd.Timestamp(end_of_dataset) - relativedelta(days=2*forecast_range_days) - datetime.timedelta(hours=(1))
       
        #check_date2 relevant if the test_set is used to make preds besides the valid_set

        end_of_dataset_time_delta = pd.Timestamp(end_of_dataset) - pd.Timestamp(start_date_valid_set)
        forecast_delta = check_date_forecast - pd.Timestamp(start_date_valid_set)
        double_forecast_delta = check_date_double_forecast - pd.Timestamp(start_date_valid_set)

        
        # 2.1.1) check if start_date_valid_set is exactly the last date of dataset:
        if ((end_of_dataset_time_delta.seconds // 3600) == 0 and end_of_dataset_time_delta.days ==0):
            if verbose > 0:
                print('# >>> Exactly last date reached! Very last single prediction is made with valid_set')    
            start_date_test_set = None
            ceiling_flag_valid = True
            ceiling_flag_test = True
            stop_preds_flag = False
        
        # 2.1.2) check if start_date_valid_set already out of range
        if end_of_dataset_time_delta.days < 0:
            #stop making predictions with current model since we already reached end of dataset:
            if verbose > 0:
                print('# >>> End of dataset reached! Stop Predictions')            
            stop_preds_flag = True
        
        # 2.1.3) check if start_date_valid_set almost reached end of dataset
        if (((forecast_delta.seconds // 3600) == 0 and forecast_delta.days ==0) or forecast_delta.days < 0) and (end_of_dataset_time_delta.days > 0 or (end_of_dataset_time_delta.days == 0 and (end_of_dataset_time_delta.seconds // 3600) >= 1)):
            #Note: there are no negative seconds in time_delta --> use "days"
            #break if we reach end of dataset: '2018-06-30 22:00:00' 
            if verbose > 0:
                print('# >> end of dataset is reached with preds of valid_set --> get last predictions with model')
                print('current valid date: ', start_date_valid_set)
                print('current test date: ', start_date_test_set)
            start_date_test_set = None
            ceiling_flag_valid = True
            ceiling_flag_test = True
            stop_preds_flag = False
            
        
        # 2.1.4) check if start_date_valid_set & test_set almost reached end of dataset
        if (double_forecast_delta.seconds // 3600) == 0 and double_forecast_delta.days == 0:
            if verbose > 0:
                print('# >> end of dataset is reached directly with preds of test_set or in next iteration --> get last predictions with model')
                print('current valid date: ', start_date_valid_set)
                print('current test date: ', start_date_test_set)        
            ceiling_flag_test = True
            stop_preds_flag = False


        # 2.1.5) check if test_set almost reached end of dataset
        if (((forecast_delta.seconds // 3600) >= 1 and forecast_delta.days == 0) or forecast_delta.days > 1) and double_forecast_delta.days < 0:
            #set flag to true --> this way test_set is set correctly such that predictions only up to '2018-06-30 22:00:00' are made
            if verbose > 0:
                print('# >> ceiling_flag_test set to "True", end of dataset is reached with preds of test_set or in next iteration')
            ceiling_flag_test = True
            stop_preds_flag = False
            

        #3) assign values and return results
        correct_dates_list = assigned_dates_list[:2] #keep dates of trainingset
        correct_dates_list.append(start_date_valid_set)
        correct_dates_list.append(start_date_test_set)
        
        return correct_dates_list, ceiling_flag_valid, ceiling_flag_test, stop_preds_flag
        
        
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    ## 1) set some params, dicts/lists for storing results..
    #set verbose:   
    if verbosity >2:
        verbose_very_high = 1
        verbose_high = 1
        verbose_mid = 1
    
    elif verbosity >1 and verbosity < 3:
        verbose_high = 1
        verbose_mid = 1
        verbose_very_high = 0
    
    elif verbosity == 1:
        verbose_mid = 1
        verbose_high = 0
        verbose_very_high = 0
        
    else:
        verbose_high = 0
        verbose_mid = 0
        verbose_very_high = 0
        

    #set dicts which store all results:
    all_MODELS_dict = {}
    all_model_RESULTS_dict = {}
    all_detected_DATES_dict = {}
    
    #set list which stores all overall avg. RMSE results: (avg. across all years and areas per prediction)
    all_RMSE_results = []
    
    #set list which stores all detected drift dates regardless of area; Fill list with very first date of regular predictions
    detected_dates_list = [pd.Timestamp(start_of_preds_date)]
    
    #create list which stores the last starting_date of the train_set used for retraining or to make predictions:
    #initialize with very verst "start_train_year" date = 2009
    last_starting_dates_train_set = [pd.Timestamp(start_train_year)]

    
    
    ## 2) set streaming data:
    
    #check for correct flags depending on detector:
    if detector_type == 'ADWIN' or detector_type == 'STEPD' or detector_type == 'EDDM':
        converted_stream_flag = True
    
    #adjust streaming_data_df if no converted df (conversion based on predictions) is used    
    if converted_stream_flag == False:
        streaming_data_df_helper = pd.DataFrame(org_ts_series.loc[start_of_preds_date:].copy()) #convert to df, if only a series was given as input

    #make time series to monitor stationary if needed:
    if converted_stream_flag == False and (use_differenced_ts == True and 'diff' in detector_type):
        print('## >> Streaming data is differenced...')
        #apply differencing scheme:
        streaming_data_df_helper = multiple_differencing(org_ts_series.copy(),differencing_scheme)
        #slice dates:
        streaming_data_df_helper = streaming_data_df_helper.loc[start_of_preds_date:]
    
    
    
    #set dataset used for training & slicing observations:
    ts_series = org_ts_series.copy()
    
    
    
    #pre_define keys of dates dict:
    area_labels = list(ts_series.columns)
    for i in range(len(area_labels)):
        if converted_stream_flag == True:
            all_detected_DATES_dict['binary{}'.format(area_labels[i])] = []
        else:
            all_detected_DATES_dict[area_labels[i]] = []
    
    
    
    #set counter for storage of results:
    counter = 0
    number_of_detector_applications = 0
    #set some mandatory parameters to start iteration:
    streaming_flag = True
    first_preds_flag = False
    
    
    '''Start Iterating'''
    
    #iterate through the streaming series:
    while streaming_flag:


        #initialize parameters & flags in very first iteration:
        if counter == 0:
            drift_flag = False
            stop_preds_flag = False
            number_of_retrainings = 0
            number_of_switch_weight_updates = 0
            retrainings_dates = []
            switch_weight_updates_dates = []
            conversion_counter = 0


            
        # 1)  ++++++++++++++++++++++++++++++++++++++++++
        
        '''Start making predictions with current model'''
              
        #1.1)
        if first_preds_flag == False:
            print('# Very first predictions are made for next {} days..'.format(first_forecast_range_days))

            first_preds_flag = True  #indicates that very first preds were made

            #make predictions for next XX days (needed to initialize detectors)               

            #assign dates to list:
            year_list = []
            year_list.append(start_train_year)
            year_list.append(last_train_set_year)
            year_list.append(start_of_preds_date) #Note: "start_of_preds_date" is updated if drift was detected in section 4.8)
            year_list.append(None)
            
            
            #double check assigned dates:
            corrected_tup = check_assigned_dates(assigned_dates_list=year_list, end_of_dataset=end_of_dataset_date, 
                                 forecast_range_days=first_forecast_range_days, verbose=verbose_high)
            
            #assign results of corrected dates and flags:
            year_list = corrected_tup[0]
            stop_preds_flag = corrected_tup[3]
            
            #only proceed if flag is set to FALSE (if set to TRUE would mean that end of dataset is reached)
            if stop_preds_flag == False:
                
                #check size of train_set: 
                #Note: necessary to distinguish if given "start_train_year" is only a whole year ('2011') or an actual timestamp ('2011-01-01 20:00:00')
                if len(str(start_train_year)) == 4 and len(str(last_train_set_year)) == 4:
                    delta_train_set = pd.Timestamp('{}-01-01'.format(pd.Timestamp(start_train_year).year)) - pd.Timestamp('{}-12-31'.format(pd.Timestamp(last_train_set_year).year))
                else:
                    delta_train_set = pd.Timestamp(start_train_year) - pd.Timestamp(last_train_set_year)

                if verbose_high > 0:
                    print(' ++ Number of days contained in train_set used for scaling: ', np.abs(delta_train_set.days))



                #get predictions with current model (currently stored in model_instance):
                #call function to make predictions:       
                model_results_dict_i, rmse_res = make_predictions_retrained_or_regular(year_list, model_instance, 
                                                                                ts_series, end_of_dataset = end_of_dataset_date,
                                                                                model_name = model_name,
                                                                                forecast_range_days = first_forecast_range_days,
                                                                                ceiling_flag_valid= corrected_tup[1], 
                                                                                ceiling_flag_test = corrected_tup[2],
                                                                                verbose=verbose_very_high)       

                
                
                if counter < 1:
                    #initialize dict with first results:
                    all_model_RESULTS_dict = model_results_dict_i.copy()

                #append remaining pred results:
                else:   
                    all_model_RESULTS_dict.update(model_results_dict_i) 
                
                #store rmse_results:
                all_RMSE_results.append(rmse_res)
                
                if verbose_mid > 0:
                    print('## Avg. RMSE of recent predictions: ')
                    print(all_RMSE_results[-1])

                
                #store very last date of preds:
                last_preds_date = pd.Timestamp(year_list[2]) + relativedelta(days=first_forecast_range_days)
                
                #increase number of times preds were made or a retraining/update was performed
                counter += 1

        
        #1.2)
        #very first preds were already made, make preds for next window...
        else:
            #make preds for next instances based on selected window:
            print('Make preds for next {} days...'.format(forecast_window_days))


            #update dates for "trainingset" & "valid set":
            start_of_next_preds_date = last_preds_date + datetime.timedelta(hours=(1)) 

            #assign dates to list:
            year_list = []
            year_list.append(start_train_year)
            year_list.append(last_train_set_year)
            year_list.append(start_of_next_preds_date)
            year_list.append(None)
            
            #double check assigned dates:
            corrected_tup = check_assigned_dates(assigned_dates_list=year_list, end_of_dataset=end_of_dataset_date, 
                                                 forecast_range_days=forecast_window_days, verbose=verbose_high)
            
            #assign results of corrected dates and flags:
            year_list = corrected_tup[0]
            stop_preds_flag = corrected_tup[3]
            
            #only proceed if flag is set to FALSE (if set to TRUE would mean that end of dataset is reached)
            if stop_preds_flag == False:
                
                #check size of train_set: 
                #Note: necessary to distinguish if given "start_train_year" is only a whole year ('2011') or an actual timestamp ('2011-01-01 20:00:00')
                if len(str(start_train_year)) == 4 and len(str(last_train_set_year)) == 4:
                    delta_train_set = pd.Timestamp('{}-01-01'.format(pd.Timestamp(start_train_year).year)) - pd.Timestamp('{}-12-31'.format(pd.Timestamp(last_train_set_year).year))
                else:
                    delta_train_set = pd.Timestamp(start_train_year) - pd.Timestamp(last_train_set_year)


                if verbose_high > 0:
                    print(' ++ Number of days contained in train_set used for scaling: ', np.abs(delta_train_set.days))



                #get predictions with current model (currently stored in model_instance):
                #call function to make predictions:       
                model_results_dict_i, rmse_res = make_predictions_retrained_or_regular(year_list, model_instance, 
                                                                                ts_series, end_of_dataset = end_of_dataset_date,
                                                                                model_name = model_name,
                                                                                forecast_range_days = forecast_window_days,
                                                                                ceiling_flag_valid= corrected_tup[1], 
                                                                                ceiling_flag_test = corrected_tup[2],
                                                                                verbose=verbose_very_high) 


                #store very last date of preds:
                last_preds_date = pd.Timestamp(year_list[2]) + relativedelta(days=forecast_window_days)
        
        
                #append prediction results to overall dict:
                #initialize dict with first results:
                if counter < 1:
                    all_model_RESULTS_dict = model_results_dict_i.copy()

                #append remaining pred results:
                else:   
                    all_model_RESULTS_dict.update(model_results_dict_i) 
                
                #store rmse_results:
                all_RMSE_results.append(rmse_res)
                
                if verbose_mid > 0:
                    print('## Avg. RMSE of recent predictions: ')
                    print(all_RMSE_results[-1])
        
                #increase number of times preds were made or a retraining/update was performed
                counter += 1
        
                  
            

        # 2)  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        '''adjusting of streaming_data_df'''
        if converted_stream_flag == True and stop_preds_flag == False:
            #2.1)
            #get current predictions and convert them to initialize streaming_data at first iteration or update streaming_data with most recent predictions:
            
            '''
            if verbose_very_high > 0:
                print('all currently available preds in dict:')
                print(all_model_RESULTS_dict)
                --> Output quite big!! Don't show preds..
            '''
            
            #call function to get all currently available predictions:
            preds_df = get_current_predictions(all_model_RESULTS_dict, verbose=verbose_very_high)
            
            #convert preds:
            converted_tuple = ddt.convert_preds_all_areas(preds_df, ts_series.copy(), verbose=verbose_high)

            if detector_type == 'ADWIN' or detector_type == 'STEPD':
                #use conversion type such that prediction errors are indicated by "0"
                converted_df = converted_tuple[0]
            else:
                #use conversion type such that prediction errors are indicated by "1"
                converted_df = converted_tuple[1]
            
            #2.2)
            #adjust streaming_data_df:
            
            #2.2.1)
            if conversion_counter < 1:
                #initialize streaming_data with first predictions:
                streaming_data_df = converted_df
            
            #2.2.2)
            else:
                #adjust current streaming_data_df:
                '''
                --> slice only most recent preds of converted_df
                --> most recent preds start at date stored in "year_list[2]" (=start_date_valid_set)
                --> "year_list[2]" is updated each time --> not necessary to distinguish between drift_flag == TRUE or FALSE
                '''
                streaming_data_df = converted_df
                #slice df such that it starts with first preds of newly trained model: (year_list[2] stores the start date of valid set)
                streaming_data_df = streaming_data_df.loc[str(year_list[2]):]
              
                
            #adjust counter:
            conversion_counter += 1
            
            #free memory:
            del converted_tuple
            del converted_df
            del preds_df

        #2.3)
        #raw data is used for streaming_data:
        if converted_stream_flag == False and stop_preds_flag == False:
            #out of "streaming_data_df_helper" slice correct dates needed for drift detection:
            #slice all dates corresponding to most recent preds:
            streaming_data_df = streaming_data_df_helper.loc[str(year_list[2]):str(last_preds_date)].copy()
            


        #2.4)
        if verbose_high > 0 and stop_preds_flag == False:
            print('## converted_stream_flag used: ', converted_stream_flag)
            print('## Shape of streaming_df: ', streaming_data_df.shape)
            print('## Head of streaming_df: ', streaming_data_df.head())
            print('## Tail of streaming_df: ', streaming_data_df.tail())

       

        
        # 3)  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        '''Apply Drift Detection'''      
        
        if stop_preds_flag == False:
            #start Drift Detection if end of dataset isn't reached
            
            print('## Start Drift Detection')

            if verbose_high > 0:
                print('# sensitivity for detection: ', sensitivity)
                print('# sensitivity_type for detection: ', sensitivity_type)


            #check how many times detectors were already applied:
            # 3.1)
            if number_of_detector_applications < 1 or drift_flag == True:
                #apply detectors the very first time:
                if verbose_high > 0:
                    print('New drift detectors applied...')
                #call function: (function returns three dicts, however only first two needed for retraining)
                detectors_dict, detected_dates_dict, _ = ddt.detectors_in_parallel(streaming_data_df, 
                                                                               detector_type, break_flag=True, 
                                                                               sensitivity=sensitivity, 
                                                                               sensitivity_type = sensitivity_type)


                #store current detectors:
                current_detectors_dict = detectors_dict


            # 3.2)
            #if no drift was found:
            if number_of_detector_applications > 0 and drift_flag == False:
                if verbose_high > 0:
                    print('current detectors are re-used since no drifts were detected...')

                #call function: (function returns three dicts, however only first two needed for retraining)
                detectors_dict, detected_dates_dict, _ = ddt.detectors_in_parallel(streaming_data_df, 
                                                                               detector_type, break_flag=True, 
                                                                               sensitivity=sensitivity, 
                                                                               sensitivity_type = sensitivity_type,
                                                                               use_predefined_detectors=True,
                                                                               predefined_detectors_dict=current_detectors_dict
                                                                              )


                #store current detectors:
                current_detectors_dict = detectors_dict


            #count number of times detectors are applied on streaming data
            number_of_detector_applications += 1


            # 3.3)
            #check for detected change in any area in dict:  
            tempy_dates_list = list(detected_dates_dict.items())
            #remove areas without any date entry:
            tempy_dates_list = [tup for tup in tempy_dates_list if tup[1]]

            #check if areas exist where drift was detected:
            if tempy_dates_list: 
                #access first entry of tuple in list:
                date_of_change = tempy_dates_list[0][1][0]
                print('Drift detected at: ', date_of_change)
                drift_flag = True

                #append date to overall dict:
                #Note: "tempy_dates_list[0][0]" stores the area label
                all_detected_DATES_dict[tempy_dates_list[0][0]].append(date_of_change)

                #append date_of_change also in list:
                detected_dates_list.append(date_of_change)

            #check if tempy_dates_list is empty: --> if empty, no drifts were detected!
            if not tempy_dates_list:
                print('> No drifts detected!')
                drift_flag = False
        

        
        # 4) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        if stop_preds_flag == False:
        
            # 4) if drift was detected, retrain model:
            if drift_flag == True:
                
                now = datetime.datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print('>> Current Time: ', dt_string)


                '''## Start Retraining'''
                '''
                -> start retraining with date at which drift was detected & previous days defined by:
                        "backshift" if Switching Scheme with Weight updates is performed
                        days = 730 (2years) if model is retrained regularly
                    
                    Example:
                    > if drift was detected in 2011, trainingset is defined as 2 years backwards based on specific date 
                      where drift was detected in 2011, 
                    > predictions are then made based on the specific date where drift was detected

                '''
                # 4.1)
                #check for special training scheme:
                if update_retrain_switch == True:

                    #compute difference of dates: (last seen Drift date and previous drift date)
                    delta_detect_dates = detected_dates_list[0] - detected_dates_list[-1]
                    #compute difference of dates: first date of last training and previous drift date 
                    delta_start_train_detect_date = last_starting_dates_train_set[-1] - detected_dates_list[-1]

                    '''update weights as long as difference of detected dates is less than Y days  AND 
                       current detected date is less than XX years apart from last retraining start_date 
                        --> model weights can be updated at most for the next X years based on last date of trainingset 

                        --> number of years (=XX) is based on "weight_update_range"
                        --> number of days (=Y) is based on "detect_dates_range_days"

                        Example:
                            weight_update_range = 4

                            e.g.: if retraining of model was performed in 2009 : 2010-12-31, weights of model are updated 
                                up to 2012-12-31, if next drift is detected in 2013, then new model is retrained


                        Note: calculation of "delta_start_train_detect_date" is based on very first date of train_set, used at last retraining


                    '''
                    # 4.1.1) #decide if weights should be updated or new model should be trained
                    if delta_detect_dates.days < -detect_dates_range_days  or delta_start_train_detect_date.days < -365*weight_update_range[0]:
                        update_weights_flag = False
                    else:
                        update_weights_flag = True


                    if verbose_mid > 0:
                        print(' ->> update_weights_flag set to "{}" , delta of drift dates: {}'.format(update_weights_flag, delta_detect_dates.days))
                        print(' >> delta of last start trainset & current drift: ', delta_start_train_detect_date.days)


                # 4.2) Assign dates based on decision if weights are updated or new model is trained           
                #list to store dates which should be later assinged to split data into training/valid/test set
                year_list_input = []


                #4.2.1) assign dates for Retraining/Creation of new model
                if update_weights_flag == False:
                    #get 2 years back in time based on detected change:
                    training_start_date = date_of_change - relativedelta(years=2)

                    #set valid_set & test_set to None since no predictions should be directly made with the new updated model 
                    start_valid_set = None
                    start_test_set = None


                #4.2.2) assign dates for updating model weights
                else:
                    #if only weights should be updated, then only most recent (new) observations are used for updating:
                    #set timestamp for beginning of training_set based on SECOND LAST detected DRIFT: --> this way only most recent observations are used for updateing weights

                    '''Note: due to the use of lagged values for training_set generation (generation of seasonal lags of up to 4*168 hours -> 4 weeks) 
                              the input dataset is shifted up to 672 hours 
                              --> therefore training_start_date is adjusted by "adding" the lags (setting the start date 673 hours before detected date)


                              !!! If the lags are changed for any reason in the get_data() function of the model instances, 
                              then the lag-number has to be adjusted for "datetime.timedelta(hours=(lag-number))" accordingly !!!!!!
                    '''   
                    #adjust for possible lag values + consider "weight_update_backshift":
                    if weight_update_backshift < 1 and update_retrain_switch == True:
                        #only consider most recent "unseen" observations based on second last detected drift
                        training_start_date = detected_dates_list[0] - datetime.timedelta(hours=(673))  
                        
                    if weight_update_backshift >= 1 and update_retrain_switch == True and adjust_lags_flag == False:
                        #consider XX years based on most recent detected drift
                        training_start_date =  date_of_change - relativedelta(years=weight_update_backshift)
                        
                    if weight_update_backshift >= 1 and update_retrain_switch == True and adjust_lags_flag == True:
                        training_start_date =  date_of_change - relativedelta(years=weight_update_backshift) - datetime.timedelta(hours=(673))
                    
                    if update_retrain_switch == False:
                        #get 2 years back in time based on detected change:
                        training_start_date = date_of_change - relativedelta(years=2)
                        
                    #set valid_set & test_set to None since no predictions should be directly made with the new updated model 
                    start_valid_set = None
                    start_test_set = None

                #4.2.3)
                if verbose_mid > 0:
                    print('## ++ previous detected dates: ', detected_dates_list)
                    print('## ++ last training dates: ', last_starting_dates_train_set)

                #delete first entry in list --> old detected date not needed anymore:
                del detected_dates_list[0]   

                #4.2.4)
                #check size of train_set:  
                delta_train_set = training_start_date - date_of_change

                if verbose_mid > 0:
                    print(' ++ Number of days contained in train_set used for scaling/retraining: ', np.abs(delta_train_set.days))


                if verbose_mid > 0:
                    print('#### Current dates: ')
                    print('#### training_start_date: ', training_start_date)
                    print('#### start_valid_set: ', start_valid_set)
                    print('#### start_test_set: ', start_test_set)


                #4.3) 
                #check if training_start_date is out of range:
                train_start_delta = pd.Timestamp(first_date_dataset) - training_start_date

                if train_start_delta.days > 0:
                    print('Year assigned < first entry in org. dataset!! ')
                    print('training_start_date: ', training_start_date)               
                    print('reassign training_start_date to very first date in data set')             
                    #reassign training_start_date to date within dataset:
                    training_start_date = pd.Timestamp(first_date_dataset)



                #4.4)
                #append dates:
                year_list_input.append(training_start_date) #append previous year
                year_list_input.append(date_of_change) #append date of detected change (last observation for training)
                year_list_input.append(start_valid_set) #used to determine start of validation set
                year_list_input.append(start_test_set) #used to determine start of test set

                                
                #4.5) create new models: either update weights or create new model
                if update_retrain_switch == True and update_weights_flag == True:

                    print('### ### Model weights are updated based on Switching Scheme')                                      

                    #call function to retrain model with currently assigned dates (NOTE: pred_results are not stored):
                    model_dict_i, _, _ = start_retraining(year_list_input, model_name, model_instance,
                                                            ts_series, XGBoost_flag = XGBoost_flag, window_step_size=window_step_size,  
                                                            index_of_start_train_year=index_of_start_train_year, 
                                                            index_of_last_train_year=index_of_last_train_year, 
                                                            update_weights_flag = update_weights_flag,
                                                            n_epochs_weight = n_epochs_weight, 
                                                            n_epochs_retrain = n_epochs_retrain,
                                                            overwrite_params = overwrite_params,
                                                            get_preds_flag = False, 
                                                            modelcounter = counter,
                                                            end_of_dataset = end_of_dataset_date,
                                                            verbose=verbose_high)


                    #count number of weight updates:
                    number_of_switch_weight_updates += 1
                    #store drift date:
                    switch_weight_updates_dates.append(date_of_change)



                else:
                    print('### ### New Model is trained')
                    #call function to retrain model:    
                    model_dict_i, _, _ = start_retraining(year_list_input, model_name, model_instance,
                                                            ts_series, XGBoost_flag = XGBoost_flag, window_step_size=window_step_size,  
                                                            index_of_start_train_year=index_of_start_train_year, 
                                                            index_of_last_train_year=index_of_last_train_year, 
                                                            update_weights_flag = update_weights_flag,
                                                            n_epochs_weight = n_epochs_weight, 
                                                            n_epochs_retrain = n_epochs_retrain,
                                                            overwrite_params = overwrite_params,
                                                            get_preds_flag = False,
                                                            modelcounter = counter,
                                                            end_of_dataset = end_of_dataset_date,
                                                            verbose=verbose_high)




                    #count number of retrainings:
                    number_of_retrainings += 1
                    #store drift date:
                    retrainings_dates.append(date_of_change)

                    #update "last_starting_dates_train_set" list, since a NEW model based on NEW training set was created
                    last_starting_dates_train_set.append(training_start_date)

                    #update "weight_update_range" if multiple ranges are used:
                    if len(weight_update_range) > 1:
                        del weight_update_range[0]
                    

                #4.6) store results
                #append results to overall dict:

                #initialize dict with first results:
                if counter < 1:
                    all_MODELS_dict = model_dict_i.copy()

                #append remaining models/results:
                else:   
                    all_MODELS_dict.update(model_dict_i)



                #increase counter
                counter +=1


                #4.7) adjust dates for next iteration:

                #use dates for training & predictions based on recent retraining of model:

                #assign dates depending on which training scheme was applied: switch update or regular retraining:            
                if make_preds_with_weight_range == True and update_retrain_switch == True and update_weights_flag == True:
                    '''
                    if "make_preds_with_weight_range" == True + switching scheme + weight updating was performed:
                    the first date of training set ("start_train_year") is set to date which was also used for weight updating 
                    
                    --> NOTE: distinction between the cases is important since, the training set specifies how data for the predictions is scaled!!

                    '''
                    if verbose_mid > 0:
                        print('Training data for weight updating is used to make predictions with updated model ')
                                        
                    start_train_year = training_start_date 
                    '''
                    #Note: instead of updating "last_starting_dates_train_set" only "training_start_date" is used 
                           --> this way to decide if the complete model should be retrained (update_weights_flag == False) 
                               during Switching Scheme is still based on "training_start_date" which was used for the last complete retraining 
                               
                               for instance:
                                   if model was completely retrained with dates: 2009-01-01 to 2010-12-31 and weights were updated up to '2011-12' based on switching scheme
                                   but if new drift is detected in '2012' --> to determine if weights should be updated or new model should be created, '2009-01-01' is used to compute "delta_start_train_detect_date"
                    
                    '''
                    last_train_set_year = date_of_change
                    start_of_preds_date = date_of_change + datetime.timedelta(hours=(1)) #start making preds for next hours..

                    
                else:
                    '''"start_train_year" is set to last starting date --> if the model was completely retrained (no weight updating with swithing scheme) 
                        then the new start date for "start_train_year"  is updated based on previous dates used for retraining
                        
                        if "make_preds_with_weight_range" == False, but switching scheme + weight updating was performed this means, 
                        that the start date for the training set, which is used to make predictions, is set back to the date at which
                        the model was complete retrained the last time --> this way the training set is "increased"
                        
                        --> NOTE: the distinction between these two cases is important since, the training set specifies how data for the predictions is scaled!!
                        
                    '''
                    start_train_year = last_starting_dates_train_set[-1]
                    last_train_set_year = date_of_change
                    start_of_preds_date = date_of_change + datetime.timedelta(hours=(1)) #start making preds for next hours..



                #4.8) adjust remaining parameters for next iteration:
                ''' 
                reset flag: --> this way again some first preds are made with the newly updated/retrained model are made 
                                before drift detectors are applied...
                '''
                first_preds_flag = False 


                print('## Predictions with retrained model are made..')

               
                print('>> Current Number of weight updates based on Switching Scheme: ', number_of_switch_weight_updates)
                print('>> Current Number of retrainings: ', number_of_retrainings)

        # 5) +++++++++++++++++++++++++++++++++++++++
        
        #adjust flag to stop while-loop if end of streaming_data_df is reached or no drift is detected and no predictions are made anymore:
        if stop_preds_flag == True:
            print('Stop streaming >> end of data set or end of predictions are reached')
            streaming_flag = False

        else:
            streaming_flag = True 


    

    print('>> Total Number of weight updates based on Switching Scheme: ', number_of_switch_weight_updates)
    print('>> Total Number of retrainings: ', number_of_retrainings)
    
    
    return  all_MODELS_dict, all_model_RESULTS_dict, all_detected_DATES_dict, all_RMSE_results, retrainings_dates, switch_weight_updates_dates



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    