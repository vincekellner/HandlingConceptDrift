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

#import custom drift detectors
from custom_drift_detectors import HDDDM
from custom_drift_detectors import STEPD
from custom_drift_detectors import MannKendall


#Import modules for drift detection:
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM


###################################### Functions for Drift Detection ####################


#calculates percentage deviation of preds and actuals and converts result into binary variable depending on specified threshold
def convert_predictions_into_binary(preds, actuals, abs_threshold = 50, perc_threshold=0.1, error_indication=0, verbose=0):
    
    #calculate percentage deviation
    abs_perc_devs = np.abs((actuals - preds)/actuals)
    #calculate absolute deviation:
    abs_dev = np.abs(actuals - preds)
    
    converted_series = pd.DataFrame()
    converted_series['percentage_dev'] = abs_perc_devs
    converted_series['abs_dev'] = abs_dev
    
    
    #create nested function:  
    def label_converter(row_perc, row_abs, abs_threshold, perc_threshold, error_indication):      
        '''
        #for each row function converts absolute percentage error into binary variable based on thresholds:
        '''

        # NOTE:  based on sk-multiflow definition: for ADWIN: "1" = means learners predictions was correct, "0" means learner's prediction was wrong
        #For EDDM however: "1" = means learner's prediction was incorrect

        error = 0
        correct = 1

        #print('row_perc, ', row_perc)
        #print('row_abs, ', row_abs)

        if error_indication == 1:
            error = 1
            correct = 0

        if row_perc > perc_threshold and row_abs > abs_threshold:  
            return error
        else:
            return correct 

    
    
    #convert percentage deviation > threshold into "1" else: "0"
    converted_series['conversion'] = converted_series.apply(lambda row: label_converter(row['percentage_dev'], row['abs_dev'], abs_threshold, perc_threshold, error_indication), axis=1)
    
    #reassign series and renaming:
    converted_series = converted_series['conversion']
    converted_series.name = 'binary' + actuals.name
    
    
    #get share of "classification errors":
    share_of_cl_error = converted_series.sum()/len(converted_series)
    
    if verbose == 1:
        
        print ('## area ', actuals.name)
        
        if error_indication == 0:

            print('Share of wrongly classified observations: ', 1 - share_of_cl_error)

        else:
            print('Share of wrongly classified observations: ', share_of_cl_error)
    
    
    #call garbage collector:
    gc.collect()
    
    return converted_series





#function converts predictions for all areas into binary & returns two converted dfs
def convert_preds_all_areas(preds_df, org_ts_df, verbose=0):
    ### Convert Predictions into Binary for all areas:
    
    if len(preds_df.shape) < 2:
        #convert single series into df:
        preds_df  = pd.DataFrame(preds_df)
    
    #create df which converts correct predictions into "1" and additional df for "0"
    for i in range(preds_df.shape[1]):

        preds = preds_df.iloc[:,i]
        #match index of preds to get correct dates:
        actuals = org_ts_df[org_ts_df.index.isin(list(preds.index))].copy()       
        #slice correct area:
        actuals = actuals.iloc[:,i]
        
        #"1" indicating correct prediction
        converted_series_error_0 = convert_predictions_into_binary(preds, actuals, perc_threshold=0.1, verbose=verbose)
        #"0" indicating correct prediction
        converted_series_error_1 = convert_predictions_into_binary(preds, actuals, perc_threshold=0.1, error_indication=1, verbose=verbose)

        if i < 1:
            converted_df_error_0 = pd.DataFrame(converted_series_error_0)
            converted_df_error_1 = pd.DataFrame(converted_series_error_1)

        else:
            converted_df_error_0 = pd.concat([converted_df_error_0,converted_series_error_0],axis=1)
            converted_df_error_1 = pd.concat([converted_df_error_1,converted_series_error_1],axis=1)


    return converted_df_error_0, converted_df_error_1



    
    
    
    
def initialize_detectors(detector_type):
    
    #note PH test uses differenced raw data! [168,24]
    #note MK_diff test uses differenced raw data! [168,24]
    #Note: HDDDM_diff actually same as "HDDDM" but important to name differently for the retrain function which looks for the "diff" term in the name
    
    detectors_dict = {'HDDDM': HDDDM(3*4*168, gamma=1.5),
                      'HDDDM_diff': HDDDM(3*4*168, gamma=1.5),
                      'STEPD': STEPD(3*4*168),
                      'MK': MannKendall(min_instances = 3*4*168, instances_step = 168, test_type = 'seasonal', alpha=0.01, period = 52, slope_threshold = 0.05),
                      'MK_diff': MannKendall(min_instances = 3*4*168, instances_step = 168, test_type = 'original_mk', alpha=0.05, slope_threshold = 0.00),
                      'ADWIN': ADWIN(delta=0.0007),
                      'PH': PageHinkley(min_instances = 3*4*168, threshold = 700, delta = 900),
                      'PH_diff': PageHinkley(min_instances = 3*4*168, threshold = 1200, delta = 1000)
                     
                     }
    
    return detectors_dict[detector_type]
    
    
    
    
    
#function applies specified drift detector on all series of given df
def detectors_in_parallel(data_df, detector_type, break_flag=True, sensitivity=1, 
                          sensitivity_type = 'monthly', use_predefined_detectors = False, 
                          predefined_detectors_dict = {}, verbose=0):
    
    '''
    function returns dates at which a drift was detected as dict{}
    -> the dict contains the area label as key and the corresponding detected dates
    
    #Params: 
    
    break_flag = indicates if functions should stop immediately if any change is detected
    detector_type = set type of detector you want to apply (ADWIN = 'ADWIN', PageHinkley = 'PH'...)
    
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
    
    #create helper to get datetime of any detected change
    data_stream_helper = data_df.copy(deep=True)
    data_stream_helper.reset_index(inplace=True)
    
    #create dict to store results
    detectors_dict = {}
    detected_dates_dict = {}

    area_labels = list(data_df.columns)  
    
    change_flag = False
    
    #check if NEW drift detectors should be used or already used predictors (this way detectors can be used which were not resetted!)
    if use_predefined_detectors == False:
        #create NEW multiple detectors for each series:
        print('new detectors are created for each area...')
        for i in range(data_df.shape[1]):

            detectors_dict[area_labels[i]] = []
            #initialize detectors for EACH area: -> important to initialize within the for loop! otherwise the same detector is applied on all series!
            detector = initialize_detectors(detector_type) #calls function
            #call function to initialize new detector 
            detectors_dict[area_labels[i]].append(detector)

            #initialize storage of dates for each area:
            detected_dates_dict[area_labels[i]] = []
    
    else:
        #use given pre-defined_detectors:
        print('pre-defined detectors are used...')
        detectors_dict = predefined_detectors_dict
        
        #initialize storage of dates for each area:
        for i in range(data_df.shape[1]):
            detected_dates_dict[area_labels[i]] = []
    
    
    #count total number of changes:
    drift_counter = 0
    
    #start "streaming" data:
    for i in range(len(data_df)):
        
        if change_flag == True:
            break
        
        #iterate through each detector:
        for u in range(data_df.shape[1]):

            #add observations to each detector:
            detectors_dict[area_labels[u]][0].add_element(data_df.iloc[i,u])

            #check if change was detected:
            if detectors_dict[area_labels[u]][0].detected_change():
                #increase counter:
                drift_counter += 1
                
                #Note: since data_df changes size if drift is detected, the index returned is always based on current data_df used in function !!! not overall time_series !!
                print('## Change detected in area {}, index: {}'.format(area_labels[u], i))
                #get date of detected change:
                detected_date = data_stream_helper[data_stream_helper.index.isin([i])]['date'].values #returns value as np.array --> actuals value can be accessed by indexing
                print('date: {}'.format(pd.Timestamp(detected_date[0])))
                #convert to Timestamp:
                detected_date = pd.Timestamp(detected_date[0])
                #append detected date to list:
                detected_dates_dict[area_labels[u]].append(detected_date) 
                
                '''Note: the following conditions have to nested within the condition above which checks whether a drift was detected or not'''
                
                #if break_flag set to True and no sensitivtiy set --> stop streaming data after first drift was detected
                if break_flag == True and sensitivity < 2:
                    change_flag = True
                    break

                
                #if sensitivity was set: 
                if break_flag == True and sensitivity >= 2 and drift_counter >= 2:
                    
                    #Note: following steps only applicable if more than 1 drift was detected
                    
                    if verbose == 1:
                        print('# start sensitivity check...')
                    
                    #get list of current detected dates and check the length and date:
                    current_dates_list = list(detected_dates_dict.values())
                    #remove all empty lists (for a lot of areas no drifts should be found so far..)
                    current_dates_list = [li for li in current_dates_list if li]
                                       
                    #convert list elements into strings (currently they are Timestamps) and sort new list:
                    #Note: dates in dict are stored in lists --> each area can have multiple dates --> we need a new list ("current_dates_new_list") to store all dates
                    current_dates_new_list = []

                    for item in current_dates_list:
                        if len(item) < 2:
                            date = str(item[0])
                            current_dates_new_list.append(date)

                        else:
                            for stamp in item:
                                date = str(stamp)       
                                current_dates_new_list.append(date)


                    current_dates_new_list.sort()
                    
                    #print(current_dates_new_list)
                    
                    if verbose == 1:
                        print('# length current_dates_list: ', len(current_dates_new_list))
                        print('current dates list: ', current_dates_new_list)
                    
                    #if length <= sensitivity level --> compare dates
                    if len(current_dates_new_list) <= sensitivity:

                        #get all unordered combinations of dates:
                        unord_pairs_dates = list(itertools.combinations(current_dates_new_list,2))

                        #set threshold based on sensitivity_type:
                        if sensitivity_type == 'monthly':
                            threshold_val = -30

                        if sensitivity_type == 'quarterly':
                            threshold_val = -3*30

                        if sensitivity_type == 'yearly':
                            threshold_val = -365
                        
                        if verbose == 1:
                            print('threshold selected: ', threshold_val)

                        #calculate delta of each pair:
                        for j in range(len(unord_pairs_dates)):

                            delta_dates = pd.Timestamp(unord_pairs_dates[j][0]) - pd.Timestamp(unord_pairs_dates[j][1])
                            
                            if verbose == 1:
                                print('date1 :', unord_pairs_dates[j][0])
                                print('date2 :', unord_pairs_dates[j][1])
                                print('delta_dates :', delta_dates)

                            #check if the detected dates are within the given time interval: --> if delta falls outside given time interval --> break
                            if delta_dates.days < threshold_val:
                                change_flag = True
                                break

                    #if len(current_dates_list) > sensitivity --> break for loop --> already more drifts detected than allowed
                    else:
                        if verbose ==1:
                            print('number of dates > sensitivity --> break')
                        
                        change_flag = True
                        break
                
     
    
    #Note: for retraining it is important to return only a single date:
    
    #return only second to last detected date: 
    if sensitivity >= 2:
        
        #get second to last item of dict:
        temp_dates_list = list(detected_dates_dict.items())
        #remove areas without any date entry:
        temp_dates_list = [tup for tup in temp_dates_list if tup[1]]
        
        if verbose == 1:
            print('temp_dates_list: ', temp_dates_list)
         
                
        #create df to sort dates:
        temp_df = pd.DataFrame()
        
        #create lists to store ALL detected dates in df with the corresponding area
        area_label_li = []
        all_dates_li = []
        
        for tup in temp_dates_list:
            if len(tup[1]) < 2:
                all_dates_li.append(tup[1][0])
                area_label_li.append(tup[0])
                
            else:
                for entry in tup[1]:
                    all_dates_li.append(entry)
                    area_label_li.append(tup[0]) 
         
        
        temp_df['area'] = area_label_li
        temp_df['date'] = all_dates_li

        #set dates as index:
        temp_df = temp_df.set_index('date')
        #sort index:
        temp_df = temp_df.sort_index()
        
        if verbose == 1:
            print('# temp_df: ')
            print(temp_df)
        
        #get second to last timestamp & area --> both are stored in a new dict:
        #Note: a dict is needed, since the retraining scheme assumes a dict is returned by the function
        temp_dict = {}
        
        #check length of temp_df: if empty then return empty dict, if length < 2 return last date..
        
        if len(temp_df) >=2:
            temp_dict[temp_df.iloc[-2].values[0]] = [temp_df.iloc[-2,:].name]

            print('Second last date of dict is returned only: ')

            #firstly return detcetors, then return second last date, but also dict with all currently detected dates:
            return detectors_dict, temp_dict, detected_dates_dict
        
        elif len(temp_df) == 1:
            temp_dict[temp_df.iloc[-1].values[0]] = [temp_df.iloc[-1,:].name]
     
            print('Last date of dict is returned only: ')
            
            #firstly return detcetors, then return the only date in df, but also dict with all currently detected dates:
            return detectors_dict, temp_dict, detected_dates_dict
        
        else:       
            print('No drifts found ')
            #firstly return detcetors, then no drifts were found, temp_df is empty --> return empty dict of detected_dates_dict
            return detectors_dict, detected_dates_dict, None
        
    else:
        #firstly return detcetors, then return current dict; no second dict available
        return detectors_dict, detected_dates_dict, None
    
    
