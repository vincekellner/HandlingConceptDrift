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




###################################### Functions for Actual Retraining ###########################################


def start_retraining(year_list, model_name, model_instance, org_ts_series, XGBoost_flag = False,
                     window_step_size=1, update_weights_flag=False, index_of_start_train_year=0, forecast_range = 12,
                     day_forecasting = False, 
                     index_of_last_train_year=1, n_epochs_weight=150, n_epochs_retrain = 150, overwrite_params = False,
                     ceiling_flag_valid = False, ceiling_flag_test = False, 
                     end_of_dataset = '2018-06-30 22:00:00', get_preds_flag = True, verbose=0):  
    
    
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
    
    day_forecasting = Flag to indicate if forecasting should be based on "forecast_range_days" 
        
    month_forecasting = Flag to indicate if forecasting should be based on "forecast_range_months" 
        
        
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
                                forecast_range, day_forecasting, verbose=0):

        ''' nested function checks if exactly one year is selected or specific Timestamp: 
            (Note: timestamp has max length of 19) >> Then function assigns years correctly which are used to slice the data_df


            if not a single year but also month or even hour is given, we need to define "start_" & "end_" date for slicing 

            Note: validation_set & test_set are always set to lenght given by "forecast_range"

        '''       
        
        if verbose > 0:
            print('>> Dates are assigned...')
            print('>date_valid: ', date_valid)
            print('>date_test: ', date_test)
            
        
        #check forecasting scheme:
        if day_forecasting == True:
            if verbose > 0:
                print('Forecasting based on days')
        else:
            if verbose > 0:
                print('Forecasting based on months')
            
            
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
            if day_forecasting == True:
                end_validation_set_year = str(helper_stamp_valid + relativedelta(days=forecast_range)) #set length of one year
            else:
                end_validation_set_year = str(helper_stamp_valid + relativedelta(months=forecast_range)) #set length of one year
                #adjust "end_validation_set_year" --> more intuitive if forecast_range exactly predictions the next months and does not consider the very first date of the next month...
                #for instance: date = 2011-01-01 00:00:00 --> +3 months = 2011-04-01 00:00:00 --> more intuitive to make preds up to '2011-03-31 23:00:00'
                end_validation_set_year = str(pd.Timestamp(end_validation_set_year) - datetime.timedelta(hours=(1)))
                
                
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
            if day_forecasting == True:
                end_test_set_year = str(helper_stamp_test + relativedelta(days=forecast_range))
            else:
                end_test_set_year = str(helper_stamp_test + relativedelta(months=forecast_range))
                #adjust "end_test_set_year" --> more intuitive if forecast_range exactly predictions the next months and does not consider the very first date of the next month...
                #for instance: date = 2011-01-01 00:00:00 --> +3 months = 2011-04-01 00:00:00 --> more intuitive to make preds up to '2011-03-31 23:00:00'
                end_test_set_year = str(pd.Timestamp(end_test_set_year) - datetime.timedelta(hours=(1)))
            
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

    model_name = model_name + '__trainsize{}__stepsize{}__p{}'.format(training_info, window_step_size, last_year_preds)                   


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
                                             forecast_range=forecast_range, day_forecasting = day_forecasting, 
                                             verbose = verbose)
       
    
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
    #Keras Model:
    if XGBoost_flag == False:
        print('Keras Model is used...')
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
    
    #XGBoost Model:
    else:
        print('XGBoost Model is used...')
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
                                                         forecast_range=forecast_range, day_forecasting = day_forecasting,
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
                                                        forecast_range=forecast_range, day_forecasting = day_forecasting,
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

    # model_dict = contains training history & model itself | model_results_dict contains predictions of newly created model for given dates | all_rmse_results contains RMSE results of preds
    return  model_dict, model_results_dict, all_rmse_results








def make_predictions_retrained_or_regular(year_list, model_instance, org_ts_series, model_name=None, 
                                          forecast_range = 12,
                                          day_forecasting = False,
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
                                forecast_range, day_forecasting, verbose=0):

        ''' nested function checks if exactly one year is selected or specific Timestamp: 
            (Note: timestamp has max length of 19) >> Then function assigns years correctly which are used to slice the data_df


            if not a single year but also month or even hour is given, we need to define "start_" & "end_" date for slicing 

            Note: validation_set & test_set are always set to length of one year!

        '''
        
        if verbose > 0:
            print('Dates are assigned...')
            print('date_valid: ', date_valid)
            print('date_test: ', date_test)
            
        
        #check forecasting scheme:
        if day_forecasting == True:
            if verbose > 0:
                print('Forecasting based on days')
        else:
            if verbose > 0:
                print('Forecasting based on months')
            
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
            if day_forecasting == True:
                end_validation_set_year = str(helper_stamp_valid + relativedelta(days=forecast_range)) #set length of one year
            else:
                end_validation_set_year = str(helper_stamp_valid + relativedelta(months=forecast_range))
                #adjust "end_validation_set_year" --> more intuitive if forecast_range exactly predictions the next months and does not consider the very first date of the next month...
                end_validation_set_year = str(pd.Timestamp(end_validation_set_year) - datetime.timedelta(hours=(1)))
            
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
            if day_forecasting == True:
                end_test_set_year = str(helper_stamp_test + relativedelta(days=forecast_range))
            else:
                end_test_set_year = str(helper_stamp_test + relativedelta(months=forecast_range))
                #adjust "end_test_set_year" --> more intuitive if forecast_range exactly predictions the next months and does not consider the very first date of the next month...
                end_test_set_year = str(pd.Timestamp(end_test_set_year) - datetime.timedelta(hours=(1)))

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
                                             forecast_range=forecast_range, day_forecasting=day_forecasting,
                                             verbose = verbose)
    
    
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

    
    return model_results_dict, all_rmse_results
    

    
    
    
def regular_retraining_scheme(model_instance, org_ts_series, model_name='model_name', XGBoost_flag = False,
                              n_epochs_retrain = 150, overwrite_params = False, 
                              n_epochs_weight = 150, update_weights_flag = False,
                              start_date_training = '2009', last_date_training = '2010', 
                              first_date_dataset = '2009-01-01 00:00:00',
                              start_of_preds_date = '2011-01-01 00:00:00',
                              end_of_dataset_date = '2018-06-30 22:00:00', 
                              forecast_range_days = 365, 
                              forecast_range_months = 12,
                              retrain_shifting_window_days = 365,
                              retrain_shifting_window_months = 12,
                              retraining_range_years = 2,
                              retraining_range_months = 24,
                              day_forecasting = False,
                              month_forecasting = True,
                              yearly_retraining_range = True,
                              retrain_shifting_window_flag_day = False,
                              first_preds_flag = False, 
                              verbosity=0):



    '''
    Function regularly retrains a given model or creates a new model based on given dates & parameters
    

    Params:

        retraining_range_years = number of years to consider for retraining --> indicates size of training set for each model
        
        retraining_range_months = number of months to consider for retraining --> indicates size of training set for each model
        
        yearly_retraining_range = Flag indicates if "retraining_range_years" should be used or not
                                    if TRUE: "retraining_range_years" is applied, else: "retraining_range_months" is applied
        
        
        retrain_shifting_window_days = number of days the window should be shifted forwards which helps to define the dates for the training set for each iteration
        
        forecast_range_days = number of days to forecast with each model
        
        
        forecast_range_months = number of months to forecast with each model
        
        retrain_shifting_window_months = number of months the window should be shifted forwards which helps to define the dates for the training set for each iteration
        
        
        day_forecasting = Flag to indicate if forecasting should be based on "forecast_range_days" 
        
        month_forecasting = Flag to indicate if forecasting should be based on "forecast_range_months" 

        retrain_shifting_window_flag_day = Flag to indicate if shifting should be based on "retrain_shifting_window_days"
                                            --> if set to TRUE: "retrain_shifting_window_days" is applied
                                            --> else: "retrain_shifting_window_months" is applied
        
        NOTE: In case "day_forecasting" AND "month_forecasting" are TRUE or both are FALSE --> always "day_forecasting" is used
        


        first_preds_flag = If FALSE indicates that predictions should be made with existing model instance in first iteration
                            if TRUE indicates that no preds are made with existing model --> directly retrain model / create new model
        

        n_epochs_weight = epochs to use for weight updating
        
        n_epochs_retrain = epochs to use for retraiing the model / creating new model
        
        overwrite_params = needed, if epochs should differ from defaul n_epochs stored in model_instance.n_epochs
        
        update_weights_flag = Flag indicates if weights are updated or new model is created
                                --> if TRUE: weights are updated



    '''
    
    
    
    #nested function to check if assigned dates are correct or already out of range:
    def check_assigned_dates(assigned_dates_list, end_of_dataset, forecast_range, day_forecasting, verbose=0):

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

        stop_preds_flag = indicates that all preds are made and no further preds can be made



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
        if day_forecasting == True:
            check_date_forecast = pd.Timestamp(end_of_dataset) - relativedelta(days=forecast_range) 
            check_date_double_forecast = pd.Timestamp(end_of_dataset) - relativedelta(days=2*forecast_range) - datetime.timedelta(hours=(1))
        else:
            check_date_forecast = pd.Timestamp(end_of_dataset) - relativedelta(months=forecast_range) 
            check_date_double_forecast = pd.Timestamp(end_of_dataset) - relativedelta(months=2*forecast_range) - datetime.timedelta(hours=(1))

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

    
    
    #++++++++++++++++++++++++++++++++++++++++++
    
    # 1]
    #set verbosity:   
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
        
    # 2]
    #set dataset to slice correct input data for training:
    ts_series_input = org_ts_series.copy()

    #set dicts which store all results:
    all_MODELS_dict = {}
    all_model_RESULTS_dict = {}

    #set list which stores all overall avg. RMSE results: (avg. across all years and areas per prediction)
    all_RMSE_results = []

    #store first date of each training set in list:
    #initialize with first given date:
    last_starting_dates_train_set = [pd.Timestamp(start_date_training)]
    
    if len(last_date_training) == 4:
        last_date_help = '{}-12-31 23:00:00'.format(pd.Timestamp(last_date_training).year)
        last_end_dates_train_set = pd.Timestamp(last_date_help)
    else:
        last_end_dates_train_set = pd.Timestamp(last_date_training)
        
    #check if "last_date_training" is out of range:
    delta_ending = pd.Timestamp(end_of_dataset_date) - last_end_dates_train_set
    if delta_ending.days < 0:
        print('Note: "last_date_training" is out of range and set to "end_of_dataset_date"')
        last_end_dates_train_set = end_of_dataset_date
        print('last_date_training: ', end_of_dataset_date)
    
    # 3]
    #assign forecast range as selected:
    if day_forecasting == True:
        print('Forecasting based on given days is used...')
        #set forecasting range & retraining window:
        forecast_range = forecast_range_days
        #reassign flag:
        month_forecasting = False
        
    if month_forecasting == True and day_forecasting == False:
        print('Forecasting based on given months is used...')
        #set forecasting range & retraining window:
        forecast_range = forecast_range_months
       
    if month_forecasting == False and day_forecasting == False:
        print('Forecasting based on given days is used since both flags FALSE...')
        day_forecasting = True
        #set forecasting range & retraining window:
        forecast_range = forecast_range_days
    
    #assign shifting window:
    if retrain_shifting_window_flag_day == True:
        print('Shifting Window based on given days is used: ', retrain_shifting_window_days)
        retrain_shifting_window = retrain_shifting_window_days
    else:
        print('Shifting Window based on given months is used: ', retrain_shifting_window_months)
        retrain_shifting_window = retrain_shifting_window_months
    
    #assign retraining range:
    if yearly_retraining_range == True:
        print('Retraining range based on years: ', retraining_range_years)
        retraining_range = retraining_range_years
    
    else:
        print('Retraining range based on months: ', retraining_range_months)
        retraining_range = retraining_range_months


    
    


    # 4]
    #set params for iteration:
    continue_preds_flag = True
    counter = 0 #counts number of iterations
    number_of_preds_existing = 0
    number_of_retrainings = 0

    '''Start Iteration'''
    while continue_preds_flag:

        #1) Make very first preds with pre-trained model
        if first_preds_flag == False:

            first_preds_flag = True

            print('## Very first predictions with given pre-defined model are made..')
            if day_forecasting == True:
                print('days to predict: ', forecast_range)
            if month_forecasting == True and day_forecasting == False:
                print('months to predict: ', forecast_range)
                            
                        
            #assign dates to list:
            year_list = []
            year_list.append(start_date_training)
            year_list.append(last_end_dates_train_set)
            year_list.append(start_of_preds_date) 
            year_list.append(None)    

            #double check assigned dates:
            corrected_tup = check_assigned_dates(assigned_dates_list=year_list, end_of_dataset=end_of_dataset_date, 
                                 forecast_range=forecast_range, day_forecasting = day_forecasting, verbose=verbose_high)

            #assign results of corrected dates and flags:
            year_list = corrected_tup[0]
            stop_preds_flag = corrected_tup[3]


            #only proceed if flag is set to FALSE (if set to TRUE would mean that end of dataset is reached)
            if stop_preds_flag == False:

                #check size of train_set: 
                #Note: necessary to distinguish if given "start_date_training" is only a whole year ('2011') or an actual timestamp ('2011-01-01 20:00:00')
                if len(str(start_date_training)) == 4 and len(str(last_date_training)) == 4:
                    delta_train_set = pd.Timestamp('{}-01-01'.format(pd.Timestamp(start_date_training).year)) - pd.Timestamp('{}-12-31'.format(pd.Timestamp(last_date_training).year))
                else:
                    delta_train_set = pd.Timestamp(start_date_training) - pd.Timestamp(last_date_training)

                if verbose_high > 0:
                    print(' ++ Number of days contained in train_set used for scaling: ', np.abs(delta_train_set.days))



                #get predictions with current model (currently stored in model_instance):
                #call function to make predictions:       
                model_results_dict_i, rmse_res = make_predictions_retrained_or_regular(year_list, model_instance, 
                                                                                ts_series_input, end_of_dataset = end_of_dataset_date,
                                                                                model_name = model_name,
                                                                                forecast_range = forecast_range,
                                                                                day_forecasting = day_forecasting,
                                                                                ceiling_flag_valid= corrected_tup[1], 
                                                                                ceiling_flag_test = corrected_tup[2],
                                                                                verbose=verbose_high)       



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



                #increase number of times preds were made or a retraining/update was performed
                counter += 1

                #count number of times preds are made with existing model:
                number_of_preds_existing += 1


        #2) Retrain Model with next train_set given & directly get preds with newly trained model
        else:
            
            print('## New model is trained and predictions are made..')
            if day_forecasting == True:
                print('days to predict: ', forecast_range)
            if month_forecasting == True and day_forecasting == False:
                print('months to predict: ', forecast_range)
                
            
            now = datetime.datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print('>> Current Time: ', dt_string)

            
            #adjust dates for next retraining:
            #adjust training set dates:
            if retrain_shifting_window_flag_day== True:
                start_date_training = last_starting_dates_train_set[-1] + relativedelta(days=retrain_shifting_window)
            else:
                start_date_training = last_starting_dates_train_set[-1] + relativedelta(months=retrain_shifting_window)
            
            if yearly_retraining_range== True:
                last_date_training =  start_date_training + relativedelta(years=retraining_range)
                
            else:
                last_date_training =  start_date_training + relativedelta(months=retraining_range)
            
            #adjust "last_date_training" to not include first date of next month..
            #for instance: start_date_training = '2009-04-01 00:00:00', "last_date_training" should equal '2011-03-31 23:00:00' not '2011-04-01 00:00:00'
            last_date_training = last_date_training - datetime.timedelta(hours=(1)) 
            
   
            #adjust remaining dates:  
            start_date_valid_set = last_date_training + datetime.timedelta(hours=(1)) 
            start_date_test_set = None #only make preds with valid_set
            

            #assign dates for retraining:
            year_list = []
            year_list.append(start_date_training)
            year_list.append(last_date_training)
            year_list.append(start_date_valid_set) 
            year_list.append(start_date_test_set) 

            #double check assigned dates:
            corrected_tup = check_assigned_dates(assigned_dates_list=year_list, end_of_dataset=end_of_dataset_date, 
                                 forecast_range=forecast_range, day_forecasting = day_forecasting, verbose=verbose_high)

            #assign results of corrected dates and flags:
            year_list = corrected_tup[0]
            stop_preds_flag = corrected_tup[3]


            #only proceed if flag is set to FALSE (if set to TRUE would mean that end of dataset is reached)
            if stop_preds_flag == False:
                
                print('# Start training new model and make predictions..')
                
                #check size of train_set: 
                #Note: necessary to distinguish if given "start_date_training" is only a whole year ('2011') or an actual timestamp ('2011-01-01 20:00:00')
                if len(str(start_date_training)) == 4 and len(str(last_date_training)) == 4:
                    delta_train_set = pd.Timestamp('{}-01-01'.format(pd.Timestamp(start_date_training).year)) - pd.Timestamp('{}-12-31'.format(pd.Timestamp(last_date_training).year))
                else:
                    delta_train_set = pd.Timestamp(start_date_training) - pd.Timestamp(last_date_training)

                if verbose_high > 0:
                    print(' ++ Number of days contained in train_set used for scaling: ', np.abs(delta_train_set.days))

                #call function for retraining:
                model_dict_i, model_results_dict_i, rmse_results_i = start_retraining(year_list, model_name, 
                                                                                      model_instance = model_instance, 
                                                                                      org_ts_series = ts_series_input,
                                                                                      XGBoost_flag = XGBoost_flag,
                                                                                      update_weights_flag=update_weights_flag, 
                                                                                      n_epochs_weight = n_epochs_weight,
                                                                                      n_epochs_retrain = n_epochs_retrain,
                                                                                      overwrite_params = overwrite_params,
                                                                                      forecast_range = forecast_range,
                                                                                      day_forecasting = day_forecasting,
                                                                                      ceiling_flag_valid = corrected_tup[1], 
                                                                                      ceiling_flag_test = corrected_tup[2],
                                                                                      end_of_dataset = end_of_dataset_date, 
                                                                                      get_preds_flag = True, 
                                                                                      verbose=verbose_high)



                if counter < 1:
                    #initialize dict with first results:
                    all_model_RESULTS_dict = model_results_dict_i.copy()
                    all_MODELS_dict = model_dict_i.copy()

                #append remaining pred results:
                else:   
                    all_model_RESULTS_dict.update(model_results_dict_i) 
                    all_MODELS_dict.update(model_dict_i)

                #store rmse_results:
                all_RMSE_results.append(rmse_results_i)

                if verbose_mid > 0:
                    print('## Avg. RMSE of recent predictions: ')
                    print(all_RMSE_results[-1])



                #increase number of times preds were made or a retraining/update was performed
                counter += 1
                
                #count number of retrainings:
                number_of_retrainings += 1
                
                
                #update "last_starting_dates_train_set" list, since a NEW model based on NEW training set was created
                last_starting_dates_train_set.append(start_date_training)
                del last_starting_dates_train_set[0]
                


            # ++++++++++++++++++++++++++++++++++++

        # 3) 
        #adjust flag to stop while-loop if end of data set is reached and no predictions are made anymore:
        if stop_preds_flag == True:
            print('Stop retraining scheme >> end of data set or end of predictions are reached')
            print(' >> Number of predictions with existing model: ', number_of_preds_existing)
            print(' >> Number of retrainings: ', number_of_retrainings)
            continue_preds_flag = False

        else:
            continue_preds_flag = True 
        
        
        
    return all_MODELS_dict, all_model_RESULTS_dict, all_RMSE_results
    

