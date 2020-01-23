import pandas as pd
import numpy as np
 

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import itertools

import os

import math

import gc

import pickle

import xgboost as xgb

#import SKlearn Wrapper:
from xgboost.sklearn import XGBRegressor

#import Class for inheritance
from basedeep import BaseDeepClass


class MultivarXGBoostWrapper(BaseDeepClass):
    
    
    def __init__(self, objective ='reg:squarederror', subsample = 0.8, colsample_bytree = 0.8, learning_rate = 0.04,
                 max_depth = 7, reg_alpha = 0.001, n_estimators = 1000, n_jobs=8, random_state = 123, verbosity=0,
                 early_stopping_rounds = 20,
                 n_timesteps=168, seasonal_lags_flag = True, early_stopping_flag = False, retraining_memory_save_mode = False,
                ):
        

        
        '''
        Variable explanation:
            >> objective = learning objective / objective function to use for learning task
            
            >> n_estimators = Number of trees to fit
            
            >> max_depth = Maximum tree depth for base learners
            
            >> n_timesteps = Number of lags used for sequences in LSTM network

            
            >> seasonal_lags_flag: Indicates whether additional features (besides regular lags) should be created 
                                    [Note: Used to be called "additional_features_flag"], 
                                    currently: if set to "True", "seasonal lags" are used or not
                                    
        
            >> learning_rate: Boosting learning_rate for training the model 
        
            
            >> reg_alpha: weight for L1 Regularization
            
            >> subsample: Subsample ratio of training instance
            
            >> colsample_bytree: Subsample ratio of columns when constructing each tree
            
            >> random_state: Rnadom number seed
            
            >> early_stopping_flag: Flag to indicate whether Early_stopping of training should be applied: 
                                    --> Trainings stops, if evaluationmetric does not improve for XX rounds 
                                                
            
            >> prediction_model: prediction_model trained with input data and given parameters
            
            >> actuals: actuals values of time series used to split data into training/valid/test sets, 
                        Note: actuals = DataFrame
                        
            >> model_name: name of model given during training
            
            >> training_history: most recent history of training    
            
        '''        

        self.n_timesteps = n_timesteps
        
        self.seasonal_lags_flag = seasonal_lags_flag
        
        self.learning_rate = learning_rate 
                        
        self.objective = objective
        
        self.n_estimators = n_estimators
        
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree 
        self.max_depth = max_depth
        
        self.reg_alpha = reg_alpha
        
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_flag = early_stopping_flag

        self.verbosity = verbosity
        
        self.prediction_model = None
        self.actuals = None
        
        self.model_name = None
        self.training_history = None
        
        self.retraining_memory_save_mode = retraining_memory_save_mode
        
     

    def get_params(self):
        '''
        Returns all parameters of model
        '''
        
        param_dict = {"n_timesteps": self.n_timesteps, 
                      "seasonal_lags_flag" : self.seasonal_lags_flag,
                      "objective" : self.objective,
                      "n_estimators" : self.n_estimators,
                      "learning_rate" : self.learning_rate,
                      "subsample" : self.subsample,
                      "colsample_bytree" : self.colsample_bytree,
                      "max_depth" : self.max_depth,
                      "reg_alpha" : self.reg_alpha,
                      "n_jobs" : self.n_jobs,
                      "random_state" : self.random_state,
                      "early_stopping_rounds" : self.early_stopping_rounds,
                      "early_stopping_flag" : self.early_stopping_flag,
                      "verbosity" : self.verbosity,
                      "model_name" : self.model_name,
                      "prediction_model" : self.prediction_model,
                      "actuals" : self.actuals
                     }
                     
        return param_dict
    
  
   
    def load_model(self, model_to_load, model_name = '_no_name_given'):
        
        '''
        function "loads" pre-trained model which was stored on disk into Class
        
        Note: this only works if params of Class are the same as params used to train "model_to_load"
        
        '''
        
        self.prediction_model = model_to_load
        self.model_name = model_name
        
        
        
    def get_multivar_feat_data_dict(self, multivar_series, seasonal_lag_set, n_timesteps):
        
        '''
        #function creates dfs for each area with taxi requests and additional features: one-hot-encoding of area labels, 
        weekofday, lags...
        #dfs are stored in dict
        '''

        multivar_series_copy = multivar_series.copy()
        #store column labels of org. data:
        area_labels = list(multivar_series_copy.columns)

        # 1) encode area_labels:
        areas_encoded_dict = self.create_one_hot_encoded_areas(multivar_series_copy)

        # 2) append weekofday encoding, hourofday encoding, monthofyear encoding, lags features:
        #Note: lags have to be appended on last step, since we drop rows with no valid lags (NaNs)

        #for each area:
        for i in range(len(area_labels)):

            #append weekofday encoded features
            weekofday_dummies_area_i = self.get_day_of_week_features_one_hot_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             weekofday_dummies_area_i],axis=1)

            #append hourofday encoding
            hourofdays_encod_area_i = self.get_hour_of_day_features_sin_cos_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             hourofdays_encod_area_i],axis=1)

            #append monthofyear encoding
            monthofyear_encod_area_i = self.get_month_of_year_features_sin_cos_encoded(multivar_series_copy.iloc[:,i])
            #append encoded features on axis = 1:
            areas_encoded_dict['area{}'.format(area_labels[i])] = pd.concat([areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                             monthofyear_encod_area_i],axis=1)


        #append lagged (seasonal lags & sliding window) and get final dataset for each area with all features:
        #-> this way we already get a "supervised" dataset very efficiently
        self.seasonal_lags_flag = True
        for i in range(len(area_labels)):
            ts_all_featrs = self.create_supervised_data_single_ts(areas_encoded_dict['area{}'.format(area_labels[i])],
                                                                  self.n_timesteps, self.seasonal_lags_flag, seasonal_lag_set)

            #assign final dataset of each area to dict:
            areas_encoded_dict['area{}'.format(area_labels[i])] = ts_all_featrs


        return areas_encoded_dict
    
    
    
    
    
    
    def generate_data(self, multivar_series, start_train_year, last_train_set_year, 
                      start_validation_set_year, start_test_set_year, 
                      end_validation_set_year=None, end_test_set_year=None, verbose=0):  
        
        
        '''
        function creates training data for model:
            several functions are called to get correct shape of data and corresponding features
        
            after that, data is split into training data/ valid data / test data based on given input dates/"years"
            
        '''

        
        if verbose == 1:
            print('generate data..')
            print('start_train_year: ', start_train_year)
            print('last_train_set_year: ', last_train_set_year)
            
            print('start_validation_set_year: ', start_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('end_test_set_year: ', end_test_set_year)
        
        
        multivar_input_series = multivar_series.copy()

        # 1) get df for each area with encoded features appended:
        lag_set = [168,336,504,672]
        areas_encoded_dict = self.get_multivar_feat_data_dict(multivar_input_series, lag_set, self.n_timesteps)


        # 2) get Train/Test-Split for each area & scale data:
        
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        if verbose == 1:
            print('# adjusted dates..')
            print('start_train_year: ', start_train_year)
            print('last_train_set_year: ', last_train_set_year)
            
            print('start_validation_set_year: ', start_validation_set_year)
            print('start_test_set_year: ', start_test_set_year)
            print('end_validation_set_year: ', end_validation_set_year)
            print('end_test_set_year: ', end_test_set_year)
            
        
        #create dict to store results:
        supervised_data_dict = {}
        #for each area create tain/test split & scale data & append to dict:
        for key in areas_encoded_dict:
            #get train/validation/test split:      
            ts_train = areas_encoded_dict[key].loc[start_train_year:last_train_set_year] 
            ts_test = areas_encoded_dict[key].loc[start_test_set_year:end_test_set_year]             
            ts_valid = areas_encoded_dict[key].loc[start_validation_set_year:end_validation_set_year]
            
                
            #append X,y pairs to dict:        
            #since we already created "supervised" data by including lags in df, we only have to slice df to receive X and y:
            X_train = ts_train.iloc[:,1:].values
            X_valid  = ts_valid.iloc[:,1:].values                   
            X_test = ts_test.iloc[:,1:].values

            y_train = ts_train.iloc[:,0].values       
            y_valid = ts_valid.iloc[:,0].values
            y_test = ts_test.iloc[:,0].values 

            #append results to dict:
            supervised_data_dict[key] = []
            supervised_data_dict[key].append(X_train)
            supervised_data_dict[key].append(y_train)
            supervised_data_dict[key].append(X_valid)
            supervised_data_dict[key].append(y_valid)
            supervised_data_dict[key].append(X_test)
            supervised_data_dict[key].append(y_test)


        #quick check if shape is correct:
        if verbose == 1:
            print('X_train shape of area237 before concat with other areas: ', supervised_data_dict['area237'][0].shape)
            print('X_valid shape of area237 before concat with other areas: ', supervised_data_dict['area237'][2].shape)
            print('X_test shape of area237 before concat with other areas: ', supervised_data_dict['area237'][4].shape)
            print('y_train shape of area237 before concat with other areas: ', supervised_data_dict['area237'][1].shape)
            print('y_valid shape of area237 before concat with other areas: ', supervised_data_dict['area237'][3].shape)
            print('y_test shape of area237 before concat with other areas: ', supervised_data_dict['area237'][5].shape)


        #create training set, valid & test set containing inputs of all selected areas: -> append all areas into one big np.array!

        #create dict to store results:
        key_list = list(supervised_data_dict.keys())

        #fill arrays with entries of first area:
        X_train, y_train = supervised_data_dict[key_list[0]][0], supervised_data_dict[key_list[0]][1]
        X_valid, y_valid = supervised_data_dict[key_list[0]][2], supervised_data_dict[key_list[0]][3]
        X_test, y_test = supervised_data_dict[key_list[0]][4], supervised_data_dict[key_list[0]][5]

        for i in range(1,len(key_list)):
            X_train = np.concatenate((X_train,supervised_data_dict[key_list[i]][0]),axis=0)
            X_valid = np.concatenate((X_valid,supervised_data_dict[key_list[i]][2]),axis=0)
            X_test = np.concatenate((X_test,supervised_data_dict[key_list[i]][4]),axis=0)
            y_train = np.concatenate((y_train,supervised_data_dict[key_list[i]][1]),axis=0)
            y_valid = np.concatenate((y_valid,supervised_data_dict[key_list[i]][3]),axis=0)
            y_test = np.concatenate((y_test,supervised_data_dict[key_list[i]][5]),axis=0)

        if verbose == 1:
            print('final concatenated shape of X_train : ', X_train.shape)
            
            
        #assign/"store" actuals    
        self.actuals = multivar_series.copy()
        
           
        #call garbage collector to free memory:
        del areas_encoded_dict
        gc.collect()
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test, supervised_data_dict



    
    
    #function that creates data for model that are only used for prediction task NOT training:
    def generate_data_get_predictions(self, multivar_series, start_train_year, last_train_set_year, 
                                      start_validation_set_year, start_test_set_year, 
                                      end_validation_set_year=None, end_test_set_year=None, verbose=0):

        #call function to create data:
        X_train, y_train, X_valid, y_valid, X_test, y_test, supervised_data_dict = self.generate_data(multivar_series,
                                                                                start_train_year,
                                                                                last_train_set_year,
                                                                                start_validation_set_year,
                                                                                start_test_set_year, 
                                                                                end_validation_set_year=end_validation_set_year,
                                                                                end_test_set_year=end_test_set_year,
                                                                                verbose=verbose)            

    
        
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        if verbose > 0:
            print('Predictions are made with current model..')
        
        
        #check if self.prediction_model is None or a new model is loaded:
        if self.retraining_memory_save_mode == True:
            
            
            if self.prediction_model == None:
            
                #load existing model from disc:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')
                #load model:
                file_to_load = Save_PATH + self.model_name + '.pickle.dat'  #self.model_name stores name of previous model!
                #load model into dict:
                prediction_model = pickle.load(open(file_to_load, "rb"))
            
            else:
                #store current model in class which is probably a pre-loaded model:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')

                final_model_name =  self.model_name + '.pickle.dat'
                file_to_save = Save_PATH + final_model_name
                #save model on disk:
                pickle.dump(self.prediction_model, open(file_to_save,"wb"))
                
                
                #delete booster of self.prediction_model to free memory:
                self.prediction_model._Booster.__del__()
                self.prediction_model = None
                
                #load model which was just stored on disk:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')
                #load model:
                file_to_load = Save_PATH + self.model_name + '.pickle.dat'  #self.model_name stores name of previous model!
                #load model into dict:
                prediction_model = pickle.load(open(file_to_load, "rb"))
                
                
        
        else:
            prediction_model = self.prediction_model
        
        #get predictions for validation data:
        valid_flag = True
        validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, start_validation_set_year, 
                                                                          end_validation_set_year, valid_flag, 
                                                                          prediction_model, multivar_series, 
                                                                          supervised_data_dict, 
                                                                          'results_{}'.format(start_validation_set_year), verbose)
        
        #prediction results test data:
        valid_flag = False
        predictions_results, rmse_results_test = self.get_preds_non_state(X_test, start_test_set_year, 
                                                                          end_test_set_year, valid_flag, 
                                                                          prediction_model, multivar_series, 
                                                                          supervised_data_dict, 'results_{}'.format(start_test_set_year),
                                                                          verbose)
        
        
        #call garbage collector to free memory:
        del X_valid
        del X_test
        del X_train
        del supervised_data_dict
        
        if self.retraining_memory_save_mode == True: 
            prediction_model._Booster.__del__()
            del prediction_model
        
        gc.collect()
        

        return validation_results, predictions_results, rmse_results_valid, rmse_results_test

       
        
            
        

    def create_model(self, X_train, y_train, X_valid, y_valid, model_name, verbose=0):

        '''
        #function creates xgBoost model & fits model to data
        '''
        
        print('## Model {} is fitted..'.format(model_name))
        
        #create model with Wrapper:
        xg_reg_model = XGBRegressor(objective =self.objective, subsample=self.subsample, colsample_bytree = self.colsample_bytree, 
                              learning_rate = self.learning_rate, max_depth = self.max_depth, reg_alpha = self.reg_alpha, 
                              n_estimators = self.n_estimators, n_jobs=self.n_jobs, random_state = self.random_state, 
                              verbosity=self.verbosity)

        #create evaluation set:
        eval_set = [(X_train, y_train),(X_valid, y_valid)]

        if self.verbosity == 1:
            verbose_flag = True
        else:
            verbose_flag = False

        #fit regressor:
        
        if self.early_stopping_flag == False:
            xg_reg_model = xg_reg_model.fit(X_train, y_train, eval_metric = 'rmse', eval_set = eval_set, verbose=verbose_flag)
        else:
            xg_reg_model = xg_reg_model.fit(X_train, y_train, early_stopping_rounds = self.early_stopping_rounds, eval_metric = 'rmse', 
                                            eval_set = eval_set, verbose=verbose_flag)
        


        history = xg_reg_model.evals_result()

        print('Fitting Model done!')
        
        
        #assign/"store" model, history & model_name:
        if self.retraining_memory_save_mode == False: 
            self.prediction_model = xg_reg_model 
            
        self.model_name = model_name
        self.training_history = history
                
        
        #store model on disk in temp-folder:
        if self.retraining_memory_save_mode == True:
            Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                         'Temp_Boosting_Models/')

            final_model_name =  model_name + '.pickle.dat'
            file_to_save = Save_PATH + final_model_name
            #save model on disk:
            pickle.dump(xg_reg_model, open(file_to_save,"wb"))
        
            #delete model to release memory:
            xg_reg_model._Booster.__del__()
            del xg_reg_model
            gc.collect()

        return history



        
        

    def get_preds_non_state(self, X_test, start_year_of_preds, 
                            end_year_of_preds, valid_flag, model, original_complete_dataset, 
                            supervised_data_dict_all_areas, model_name, verbose=0):

        '''
        #get predictions for multivarXGBoost model:

        '''


        #set years correctly:
        if end_year_of_preds == None:
            end_year_of_preds = start_year_of_preds


        # 1) get predictions based on sequence input & external input:
        yhat_s = model.predict(X_test) 

        '''Note: XGBoost returns "yhat_s " as a list!! '''

        #print('yhat_s ', yhat_s)


        # 2) set index to select either validation or test set:
        if valid_flag == True:
            index_to_access = 2
        else:
            index_to_access = 4

        # 3) store predictions for each area
        yhat_all_list = []

        #get keys of dict:
        key_list = list(supervised_data_dict_all_areas.keys())

        #prepare indices to access and store rescaled predictions:   
        start_idx = 0
        end_idx = 0

        #rescale predictions for each area:
        for i in range(len(key_list)):

            #print('access ', key_list[i])

            #set parameters to access correct number of values of each area:
            end_idx += supervised_data_dict_all_areas[key_list[i]][index_to_access].shape[0]
            number_of_samples = supervised_data_dict_all_areas[key_list[i]][index_to_access].shape[0]
            '''#access yhat_s at right index for each area -> since X_test contains inputs of each area next to each other
                 we only need to access the right index..'''
            predictions_to_access = yhat_s[start_idx:end_idx] 

            #print('predictions_to_access before reshaping: ', predictions_to_access)
            #print('end_idx ', end_idx)
            #print('en(predictions_to_access) ' , len(predictions_to_access))


            #reshape data for each area:
            predictions_to_access = np.array(predictions_to_access).reshape(number_of_samples,1)

            #append results:
            yhat_all_list.append(predictions_to_access)

            #update index:
            start_idx = end_idx

        '''#create numpy_array based on yhat_rescaled_all_list: 
            (this way we have all rescaled predictions for each area in one big numpy array -> each column equals an area)'''
        yhat_all_areas = yhat_all_list[0]
        for i in range(1,len(yhat_all_list)):
            yhat_all_areas = np.concatenate((yhat_all_areas,yhat_all_list[i]),axis=1)


        #get time index of data & correct column names:
        predictions_all = pd.DataFrame(yhat_all_areas)
        predictions_all.index = original_complete_dataset[start_year_of_preds:end_year_of_preds].index
        predictions_all.columns = original_complete_dataset.columns

        if verbose == 1:
            print('predictions preview:')
            print(predictions_all.head(3))


        #get rmse:
        rmse_per_ts = []
        for u in range(predictions_all.shape[1]):
            rmse_single_ts = np.sqrt(mean_squared_error(original_complete_dataset.loc[start_year_of_preds:end_year_of_preds].iloc[:,u],
                                                        predictions_all.iloc[:,u]))
            rmse_per_ts.append(rmse_single_ts)

            if verbose == 1:
                print('RMSE per TS {} : model: {} : {}'.format(u, self.model_name, rmse_per_ts[u]))

        #get average of all rmses
        total_rmse = np.mean(rmse_per_ts)

        if verbose == 1:
            print('Avg.RMSE for model {} : {}'.format(model_name, total_rmse))


        #return RMSE results:
        rmse_results = []
        rmse_results.append(total_rmse)
        rmse_results.append(rmse_per_ts)
        
        
        #free memory:
        del original_complete_dataset
        del supervised_data_dict_all_areas
        del X_test
        del model
        
        gc.collect()


        return predictions_all, rmse_results

    
        

    def create_full_pred_model(self, multivar_series, start_train_year, last_train_set_year, 
                               start_validation_set_year, start_test_set_year, model_name, 
                               end_validation_set_year=None, end_test_set_year=None, get_preds_flag = True,
                               insert_custom_training_data_flag = False, custom_training_data_tuple = None,
                               custom_start_valid_date = None, custom_end_valid_date = None, 
                               custom_start_test_date = None, custom_end_test_date = None, 
                               verbose = 0):
        
        
        '''
        #function to preprocess data, fit model and get predictions for valid & test set "automatically" 
        '''
        
        #1)
        #get data for model:
        if insert_custom_training_data_flag == False and not custom_training_data_tuple:
            X_train, y_train, X_valid, y_valid, X_test, y_test, supervised_data_dict = self.generate_data(multivar_series,
                                                                                        start_train_year,
                                                                                        last_train_set_year,
                                                                                        start_validation_set_year,
                                                                                        start_test_set_year, 
                                                                                        end_validation_set_year=end_validation_set_year,
                                                                                        end_test_set_year=end_test_set_year,
                                                                                        verbose=verbose)                
        
        if insert_custom_training_data_flag == True and not custom_training_data_tuple:
            print('## No tuple given for training data !! ')
            print(' "custom_training_data_tuple" is empty ')
            print(' >> Stop process')
            
            return None
        
        if insert_custom_training_data_flag == True and custom_training_data_tuple:
            print('## Custom training data is used...')
            
            #assign custom data:
            '''NOTE: Only possible if tuple matches exactly the same specs than "regular training data" generated by "generate_data()"'''
            if len(custom_training_data_tuple) < 8:
                print('Given tuple does not match with expected params!!')
                print('Given length: ', len(custom_training_data_tuple))
                print('>> Stop process!! ')
                return None
            
            else:
                #assign data:
                X_train = custom_training_data_tuple[0]
                y_train = custom_training_data_tuple[1]
                X_valid = custom_training_data_tuple[2] 
                y_valid = custom_training_data_tuple[3] 
                X_test = custom_training_data_tuple[4]
                y_test = custom_training_data_tuple[5]
                supervised_data_dict = custom_training_data_tuple[-1]
                
                start_validation_set_year = custom_start_valid_date
                end_validation_set_year = custom_end_valid_date
                start_test_set_year = custom_start_test_date
                end_test_set_year = custom_end_test_date
        
        #2)
        #create model:
        print('create Multivar XGBoost Model:')       
        history_model = self.create_model(X_train, y_train, X_valid, y_valid, model_name)
        
        
        #3) get preds
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
            
        
        
        #set self.prediction_model to None:
        if self.retraining_memory_save_mode == True:
            
            self.prediction_model = None #this way the newly trained model is loaded from disc in "generate_data_get_predictions()"
        
            #load model from disc to get preds:
            Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                         'Temp_Boosting_Models/')
            #load model:
            file_to_load = Save_PATH + model_name + '.pickle.dat'

            #load model:
            prediction_model = pickle.load(open(file_to_load, "rb"))
        
        else:
            prediction_model = self.prediction_model
        
        
        if get_preds_flag == True:
            #get predictions for validation data:
            valid_flag = True
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, start_validation_set_year, 
                                                                              end_validation_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, 
                                                                              supervised_data_dict, 
                                                                              'results_{}'.format(start_validation_set_year),  
                                                                              verbose)

            #prediction results test data:
            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, start_test_set_year, 
                                                                              end_test_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, 
                                                                              supervised_data_dict,
                                                                              'results_{}'.format(start_test_set_year),  
                                                                              verbose)

            #call garbage collector to free memory:
            del X_valid
            del X_test
            del X_train
            del supervised_data_dict
            gc.collect()
        
            return (history_model, prediction_model, predictions_results, model_name, multivar_series, validation_results,
                    rmse_results_test, rmse_results_valid)

        else:
            
            #call garbage collector to free memory:
            del X_valid
            del X_test
            del X_train
            del supervised_data_dict
            gc.collect()
            
            print('## Only training history & model are returned')
            return (history_model, prediction_model)
    
        
        
        
    def retrain_model(self, multivar_series, start_train_year, last_train_set_year, 
                      start_validation_set_year, start_test_set_year, model_name, 
                      end_validation_set_year=None, end_test_set_year=None, 
                      n_estimators = 1000, max_depth = 7, n_jobs=8, overwrite_params = False, 
                      get_preds_flag=True, verbose=0):
        
                     
        '''
        #function creates new model and discards existing one --> function only calls "create_full_pred_model()"
        n_estimators & max_depth can be set if "overwrite_params" = True
        '''
        
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.n_jobs = n_jobs
            
        
        print('## New Model is created, old model is discarded..')
               
        #create new MLP model:
        results_i = self.create_full_pred_model(multivar_series, start_train_year, last_train_set_year,
                                                start_validation_set_year, start_test_set_year, model_name, 
                                                end_validation_set_year=end_validation_set_year, 
                                                end_test_set_year=end_test_set_year, get_preds_flag = get_preds_flag,
                                                verbose=verbose)
        
        #call garbage collector to free memory:
        gc.collect()
        
        #returns results as tuple:
        return results_i

    
    

    
    def update_model_weights(self, multivar_series, start_train_year, last_train_set_year, start_validation_set_year,
                             start_test_set_year, model_name, end_validation_set_year=None, end_test_set_year=None, 
                             n_estimators = 1000, max_depth = 7, n_jobs = 8, verbosity = 1, model_to_update = None, 
                             overwrite_params = False,
                             get_preds_flag = True, verbose=0):
        
        '''
        #function updates weights of exisiting model by fitting model on new data
        '''

        
        print('## Existing Model is updated..')
        
        #assign chosen parameters:
        if overwrite_params == True:
            print('#params are overwritten')
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.n_jobs = n_jobs
            self.verbosity = verbosity
        
        # 1) create data for model to be updated: 
        '''#Note: Only X_train & y_train are actually needed for updating the weights. 
                    X_valid and X_test are directly used to get predictions for these years'''
        X_train, y_train, X_valid, y_valid, X_test, y_test, supervised_data_dict = self.generate_data(multivar_series,
                                                                                        start_train_year,
                                                                                        last_train_set_year, 
                                                                                        start_validation_set_year,
                                                                                        start_test_set_year, 
                                                                                        end_validation_set_year=end_validation_set_year,
                                                                                        end_test_set_year=end_test_set_year,
                                                                                        verbose=verbose)                


        # 2) access model to be updated:
        if model_to_update:
            print('loaded model is used..')
            #if there is an existing model saved on disk that should be updated, "load" model into class:
            # !!!! Note: "loaded" model must have identical params as the Class currently has !!
            self.prediction_model = model_to_update
            
        prediction_model = self.prediction_model
        '''NOTE: "callback list" currently not available for updating the model!! 
                    (would require to copy all callbacks into this function...)'''
        
        
        #compile model:    
        '''
        #create model with Wrapper:
        xg_reg = XGBRegressor(objective =self.objective, subsample=self.subsample, colsample_bytree = self.colsample_bytree, 
                              learning_rate = self.learning_rate, max_depth = self.max_depth, reg_alpha = self.reg_alpha, 
                              n_estimators = self.n_estimators, n_jobs=self.n_jobs, random_state = self.random_state, 
                              verbosity=self.verbosity)

        '''
        #create evaluation set:
        eval_set = [(X_train, y_train),(X_valid, y_valid)]
             
        if self.verbosity == 1:
            verbose_flag = True
        else:
            verbose_flag = False
        
        
        #fit regressor:
        
        if self.retraining_memory_save_mode == True:
            
            if self.prediction_model == None:
                print('>>save mode used & self.predicion_model = None')
                    
                #load existing model from disc:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')
                #load model:
                file_to_load = Save_PATH + self.model_name + '.pickle.dat'  #self.model_name stores name of previous model!
                #load model into dict:
                prediction_model = pickle.load(open(file_to_load, "rb"))

            else:
                print('>>save mode used & self.predicion_model != None')
                #1)
                #store current model in class which is probably a pre-loaded model:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')

                final_model_name =  self.model_name + '.pickle.dat'
                file_to_save = Save_PATH + final_model_name
                #save model on disk:
                pickle.dump(self.prediction_model, open(file_to_save,"wb"))
                
                #2)
                '''
                #delete model and then reload model to free memory:
                prediction_model_old = self.prediction_model 
                
                #set self.prediction_model to None:
                self.prediction_model = None #this way the newly trained model is loaded from disc in "generate_data_get_predictions()"
                
                #delete model to release memory:
                prediction_model_old._Booster.__del__()
                del prediction_model_old
                gc.collect()
                '''
                #delete model to release memory:
                #set self.prediction_model to None:
                self.prediction_model._Booster.__del__()               
                self.prediction_model = None #this way the newly trained model is loaded from disc in "generate_data_get_predictions()"               

                
                
                #3)
                print('>> prediciton model is loaded from disk')
                print('Model to load: ', self.model_name )
                #load existing model from disc:
                Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                             'Temp_Boosting_Models/')
                #load model:
                file_to_load = Save_PATH + self.model_name + '.pickle.dat'  #self.model_name stores name of previous model!
                #load model into dict:
                prediction_model = pickle.load(open(file_to_load, "rb"))
                
                print('model is loaded from disk :', prediction_model)
                
                
        #a model is loaded or exists in class and no special mode is used:       
        else:
            print('>>save mode not used & self.predicion_model != None')
            prediction_model = self.prediction_model
            
        
        #FIT Model / Update model: 
        
        #get current underlying booster obj:       
        current_booster_obj = prediction_model.get_booster()
        print('Multivar XGBoost Model is updated..')
        updated_model = prediction_model.fit(X_train, y_train, xgb_model = current_booster_obj, eval_metric = 'rmse', 
                                                  eval_set = eval_set, verbose=verbose_flag)
        
                
        #use existing model! 
        '''
        #get current underlying booster obj:       
        current_booster_obj = self.prediction_model.get_booster()
        print('Multivar XGBoost Model is updated..')
        updated_model = self.prediction_model.fit(X_train, y_train, xgb_model = current_booster_obj, eval_metric = 'rmse', 
                                                  eval_set = eval_set, verbose=verbose_flag)
        '''
        
        history = updated_model.evals_result()
        
        #assign/"store" history & model_name:
        #self.prediction_model = updated_model
        self.model_name = model_name #update model name
        self.training_history = history
        
        if self.retraining_memory_save_mode == True:
            #store model on disk in temp-folder:
            Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                         'Temp_Boosting_Models/')

            final_model_name =  model_name + '.pickle.dat' 
            file_to_save = Save_PATH + final_model_name
            #save model on disk:
            pickle.dump(updated_model, open(file_to_save,"wb"))


            #delete model to release memory:
            updated_model._Booster.__del__()
            del updated_model
            gc.collect()
            
            self.prediction_model = None #this way the newly trained model is loaded from disc in "generate_data_get_predictions()" 
            
        else:
            self.prediction_model = updated_model
        
        
        # 3) get predictions with updated model:
        
        #set years correctly:
        if end_validation_set_year == None:
            end_validation_set_year = start_validation_set_year
        
        if start_test_set_year == None:
            start_test_set_year = start_validation_set_year
            
        if end_test_set_year == None:
            end_test_set_year = start_test_set_year
        
        
        if self.retraining_memory_save_mode == True:
            #load model from disc to get preds:
            Save_PATH = ('/media/vincent/harddrive/ML-Projects_all/NY_Cab_Project/NY_Cab_Data/results/xg_boost_Models/'
                         'Temp_Boosting_Models/')
            #load model:
            file_to_load = Save_PATH + model_name + '.pickle.dat'

            #load model into dict:
            prediction_model = pickle.load(open(file_to_load, "rb"))

        else:
            prediction_model = self.prediction_model
            
        
        if get_preds_flag == True:
            #get predictions for validation data:
            valid_flag = True
            validation_results, rmse_results_valid = self.get_preds_non_state(X_valid, start_validation_set_year, 
                                                                              end_validation_set_year, valid_flag, 
                                                                              prediction_model, multivar_series, supervised_data_dict,
                                                                              'results_{}'.format(start_validation_set_year), verbose)

            #prediction results test data:
            valid_flag = False
            predictions_results, rmse_results_test = self.get_preds_non_state(X_test, start_test_set_year, 
                                                                              end_test_set_year, valid_flag,
                                                                              prediction_model, multivar_series, supervised_data_dict,
                                                                              'results_{}'.format(start_test_set_year), verbose)


            #create results tuple with only prediction results & model: (history does not contain information about validation data!!)
            results_i = (history, prediction_model, predictions_results, model_name, multivar_series,
                         validation_results, rmse_results_test, rmse_results_valid)

        
        else:
            print('## Only training history & model are returned')
            results_i = (history, prediction_model)
        
        
        #call garbage collector to free memory:
        del X_valid
        del X_test
        del X_train
        del supervised_data_dict
        gc.collect()
        
        #returns results as tuple:
        return results_i
    
