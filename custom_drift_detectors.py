import numpy as np
import scipy.stats as st

class STEPD(object):
    
    '''
    Implementation of the Drift Detection algorithm based on "Detecting Concept Drift Using Statistical Testing" (Nishida, Yamauchi 2007)
    '''
    
    def __init__(self, recent_window = 30, alpha_w = 0.05, alpha_d = 0.003):
        
        '''
        
        recent_window = Windowsize of most recent predictions used for drift detection
        alpha_w = Significance level of a warning
        alpha_d = Significance level of a drift
        
        '''
        
        #initialize parameters:
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        
        self.r0 = 0
        self.rR = 0
        self.n0 = 0
        self.nR = recent_window
        self.p_hat = None
        self.retrain_memory = []
        self.pred_memory = []
        self.test_statistic = None
        
        self.in_concept_change = None
        self.in_warning_zone = None
        
        #self.reset()
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False
        self.in_warning_zone = False
        
        self.r0 = 0
        self.rR = 0
        self.n0 = 0
        self.retrain_memory = []
        self.pred_memory = []
        self.test_statistic = None
        self.p_hat = None
        
        # self.__init__(recent_window = self.recent_window, alpha_w = self.alpha_w, alpha_d = self.alpha_d)
        
    
    def add_element(self, prediction):
        '''
        Add new element to the statistic
        
        
        correct classification is indicated with prediction = "1"
        
        '''
        
        if self.in_concept_change:
            self.reset()
        
       
        self.pred_memory.append(prediction)
                    
        #start drift detection if n0 + nR >= 2W
        if len(self.pred_memory[:-self.nR]) >= self.nR:
            self.r0 = sum(self.pred_memory[:-self.nR])
            self.rR = sum(self.pred_memory[-self.nR:])
            self.n0 = len(self.pred_memory[:-self.nR])          
        
            #calculate test statistic:
            self.p_hat = (self.r0 + self.rR) / (self.n0 + self.nR)

            self.test_statistic = np.abs((self.r0/self.n0) - (self.rR/self.nR)) - 0.5*((1/self.n0) + (1/self.nR))
            self.test_statistic = self.test_statistic / np.sqrt(self.p_hat*(1-self.p_hat)*((1/self.n0) + (1/self.nR)))

        
            #get p_value based on gaussian standard normal distribution:
            p_value = 1-st.norm.cdf(abs(self.test_statistic))
            #p_value = st.norm.pdf(abs(self.test_statistic))
                        
                        
            if p_value < self.alpha_w and p_value < self.alpha_d:
                self.in_concept_change = True
        
            elif p_value < self.alpha_w:
                self.in_warning_zone = True
                #append predictions_index in warning zone
                prediction_index = len(self.pred_memory) - 1
                self.retrain_memory.append(prediction_index)
                
            else:
                self.in_warning_zone = False
                self.in_concept_change = False
                
                #remove instances in memory:
                self.retrain_memory = []
        
        
        
    def detected_change(self):

        return self.in_concept_change 



    def detected_warning_zone(self):

        return self.in_warning_zone


    def get_retrain_memory(self):
        #Returns the instances which satisfy p_value < warning level
        return self.retrain_memory
        
        
        






import numpy as np
import scipy.stats as st
import pymannkendall as mk

class MannKendall(object):
    
    def __init__(self, min_instances = 30, instances_step = 1, alpha = 0.05, slope_threshold = 0.0, test_type = 'original_mk', period = 12):
        
        '''
        
        min_instances = minimum instances to be considered before MK test is applied 
        
        instances_step = after minimum instances is reached, frequency MK test is applied  --> speeds up test significantly if test is not applied every single instance
                         >> "1" = test is applied for every instance
                         >> "10" = test is applied every 10th instance
        
        alpha = Significance level of test
        

        test_type = Type of Test used to perform trend detection:
        
        six different tests available:
        
            - 'original_mk'                    --> Original MK test:  Assumption: No temporal relation in data 
            - 'hamed_rao_mod'                  --> Hamed and Rao Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'yue_wang_mod                    --> Yue and Wang Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'trend_free_pre_whitening_mod'   --> Trend Free Pre Whitening Modification MK test:  Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'pre_whitening_mod'              --> Pre Whitening Modification MK test:  Assumption: Assumption: temporal relation in data (signf. autocorrelation present for lag > 1)
            - 'seasonal', period parameter needed! --> Seasonal MK test:  Assumption: temporal relation in data + seasonality 
            
        period = sesonality pattern in dataset -> "12" = monthly, "52" = weekly
        
        '''
        
        #initialize parameters:        
        self.min_instances = min_instances
        self.alpha = alpha
        self.test_type = test_type
        self.period = period
        self.instance_memory = []
        self.slope_threshold = slope_threshold
        self.instances_step = instances_step
        
        self.in_concept_change = False
        
        self.trend = None
        self.p_value = None
        self.sens_slope = 0.0
        self.sample_count = 0
        self.instance_count = 0
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False        
        self.instance_memory = []
        
        self.trend = None
        self.p_value = None
        self.sens_slope = 0.0
        self.sample_count = 0
        self.instance_count = 0

        # self.__init__(recent_window = self.recent_window, alpha_w = self.alpha_w, alpha_d = self.alpha_d)
        
    
    def add_element(self, value):
        
        '''
        Add new element to the statistic
                
        '''
        
        #reset parameters if change was detected:
        if self.in_concept_change:
            self.reset()
        
        
        
        #append elements:
        self.instance_memory.append(value)
        
                    
        
        if len(self.instance_memory) == self.min_instances:
            self.sample_count = 1
        
        if len(self.instance_memory) > self.min_instances:
            self.instance_count += 1
            
        #start drift detection: >> min_instances have to be reached, then always perform test once, after that perform test every i_th instance (instances_step)
        if len(self.instance_memory) >= self.min_instances and ((self.instance_count == self.instances_step) or (self.sample_count == 1)):
            
            if self.test_type == 'original_mk':
                
                #call corresponding test from package:
                results_tuple = mk.original_test(self.instance_memory, self.alpha)
    
            
            if self.test_type == 'hamed_rao_mod':
                
                #call corresponding test from package:
                results_tuple = mk.hamed_rao_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'yue_wang_mod':
                
                #call corresponding test from package:
                results_tuple = mk.yue_wang_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'trend_free_pre_whitening_mod':
                
                #call corresponding test from package:
                results_tuple = mk.trend_free_pre_whitening_modification_test(self.instance_memory, self.alpha)
            
            if self.test_type == 'pre_whitening_mod':
                
                #call corresponding test from package:
                results_tuple = mk.pre_whitening_modification_test(self.instance_memory, self.alpha)
                
            if self.test_type == 'seasonal':
                
                #call corresponding test from package:
                results_tuple = mk.seasonal_test(self.instance_memory, period = self.period, alpha = self.alpha)
            
            
            #reset counter every time a test was performed:
            self.sample_count = 0
            self.instance_count = 0
            
            
            #assign results:
            self.p_value = results_tuple[2]
            self.sens_slope = results_tuple[-1]
            self.trend = results_tuple[0]  
                
                        
            if self.p_value < self.alpha and np.abs(self.sens_slope) > self.slope_threshold:
                self.in_concept_change = True
                   
            else:
                self.in_concept_change = False
    
        
        
        
    def detected_change(self):

        return self.in_concept_change 


    
    def get_test_results(self):
        
        test_results = (self.trend, self.p_value, self.sens_slope)

        return test_results
    
        
        


import numpy as np
import math

class HDDDM(object):
    
    '''
    Implementation of the Drift Detection algorithm based on Hellinger Distance of Histograms (Ditzler, Polikar 2011)
    '''
    
    def __init__(self, batch_size = 30, gamma = 1.5):
        
        '''
        batch_size = number of instances in current batch presented to the algorithm

        
        '''
        
        #initialize parameters:
        self.batch_size = batch_size     
        self.hist_P = []
        self.hist_Q = []
        self.instance_memory = []
                
        self.init_flag = False
        
        self.hellinger_t = None
        self.hellinger_t_1 = 0
        self.epsilon = None
        self.epsilon_memory = []
        self.gamma = gamma
    
        self.in_concept_change = None
        
        
    def reset(self):
        '''
        reset parameters of change detector
        '''
        self.in_concept_change = False
        
        self.epsilon = None    
        #only keep epsilon of last detected change:
        self.epsilon_memory = [self.epsilon_memory[-1]]
                
        self.hist_Q = self.hist_P
        self.hist_P = []
        

    
    def discrete_hellinger_distance(self, p,q):
        '''
        currently not used
        '''
        list_of_squares = []
        for p_i, q_i in zip(p,q):
            
            #square of difference of ith element:
            s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2
            
            #append
            list_of_squares.append(s)
    
        #get sum of squares:
        sum_sq = sum(list_of_squares)
        
        return math.sqrt(sum_sq/2)
    
    
    def hist_hellinger_distance(self, hist_Q, hist_P, n_bins):

        freqs_memory_Q = []
        freqs_memory_P = []

        hist_result_Q = np.histogram(hist_Q, bins=n_bins)
        hist_result_P = np.histogram(hist_P, bins=n_bins)

        hist_sum_Q = sum(hist_result_Q[0])
        hist_sum_P = sum(hist_result_P[0])


        for i in range(len(hist_result_Q[0])):

            hist_freq_Q = np.sqrt((hist_result_Q[0][i]/hist_sum_Q))
            hist_freq_P = np.sqrt((hist_result_P[0][i]/hist_sum_P))

            freqs_memory_Q.append(hist_freq_Q)
            freqs_memory_P.append(hist_freq_P)


        hell_distance = np.sqrt(np.sum((np.array(freqs_memory_P) - np.array(freqs_memory_Q))**2))


        return hell_distance

            
    
    def add_element(self, value):
        '''
        Add new element to the statistic and create batches.
        If batch size is reached, compute Hellinger Distance based on Ditzler and Polikar 2011
                
        '''
        
        if self.in_concept_change:
            self.reset()
        
        
        #append new instances 
        self.instance_memory.append(value)      
        
        #initialize hist_Q in first iteration:
        if len(self.instance_memory) == self.batch_size and not self.init_flag:
            self.hist_Q = self.instance_memory
            
            #empty list:
            self.instance_memory = []
            
            self.init_flag = True
           
        
        #initialize hist_P:
        if len(self.instance_memory) == self.batch_size and self.init_flag:
            self.hist_P = self.instance_memory
            
            #empty list:
            self.instance_memory = []
            
            

        if len(self.hist_P) == self.batch_size:
            
        
            #compute Hellinger Distance:
            n_bins = math.floor(np.sqrt(len(self.hist_P))) #based on cardinality of self.hist_P
            self.hellinger_t = self.hist_hellinger_distance(self.hist_Q, self.hist_P, n_bins)
            
            #compute measures to update threshold:
            mean_eps = np.mean(np.abs(self.epsilon_memory))
            std_eps = np.std(np.abs(self.epsilon_memory))

            #get difference in divergence:
            self.epsilon = self.hellinger_t - self.hellinger_t_1
            
            #append epsilon:
            self.epsilon_memory.append(self.epsilon)

            #compute threshold:
            beta_t = mean_eps + self.gamma*std_eps
                       
            #update hellinger distance at (t-1)
            self.hellinger_t_1 = self.hellinger_t

            
            if abs(self.epsilon) > beta_t:
                self.in_concept_change = True

            else:
                #update hist_Q:
                self.hist_Q = self.hist_Q + self.hist_P 
                #empty self.hist_P:
                self.hist_P = []
                    
        
        
    def detected_change(self):

        return self.in_concept_change 



        
        
        
        




















