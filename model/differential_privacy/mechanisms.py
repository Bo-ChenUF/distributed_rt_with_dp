# privacy mechanism class is an abstract class
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
from abc import ABC, abstractmethod

#from util.logging_util import init_logger
#logger = init_logger()

class privacyPara:
    def __init__(self, epsilon, sensitivity, delta=0.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

class privacyMech(ABC):
    '''
    Abstract class for privacy mechanisms
    '''

    def __init__(self) -> None:
        pass
    
    def sample_one(self, data, privacyPara):
        '''
        Draw one sample from the privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data, recommended in List type.
            - privacyPara: privacy parameter object.
        
        Return:
            - Samples
        '''
        # convert input to numpy array
        try:
            data_nparray = np.array(data)
        except:
#            logger.error(f"Input: {data} can not be converted to numpy array!")
            raise ValueError("Input data can not be converted to numpy array!")

        # TODO Add sample class
        distribuition_para = self.computeDistParameter(privacyPara)
        return self.sample_one_with_distribution_para(data_nparray, distribuition_para)
    
    def sample_many(self, data, privacyPara, n: int):
        '''
        Draw n sample(s) from the privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - privacyPara: privacy parameter object.
            - n: number of samples.
        
        Return:
            - Samples
        '''
        distribuition_para = self.computeDistParameter(privacyPara)
        return self.sample_many_with_distribution_para(data, distribuition_para, n)
    
    def sample_many_with_distribution_para(self, data, distribuition_para, n):
        '''
        Draw n sample(s) from the privacy mechanism with specific distribution parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - distribuition_para: distribution parameter.
            - n: number of samples.
        
        Return:
            - Samples
        '''
        simples = []

        for _ in range(n):
            simple = self.sample_one_with_distribution_para(data, distribuition_para)
            simples.append(simple)

        return simples
    

    @abstractmethod
    def sample_one_with_distribution_para(self, data, distribuition_para):
        '''
        Return one sample with given distribution parameter(s)
        '''
        pass

    @abstractmethod
    def computeDistParameter(self, privacyPara):
        '''
        Compute privacy distribution parameters with given privacy parameters
        '''
        pass


class boundedMech(ABC):
    "Abstract class for bounded privacy mechanisms"

    def __init__(self) -> None:
        pass
    
    def sample_one(self, data, privacyPara, lb, ub):
        '''
        Draw one sample from the bounded privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data, recommended in List type.
            - privacyPara: privacy parameter object.
            - lb: lower bound for data.
            - ub: upper bound for data.
        
        Return:
            - Samples
        '''
        
        try:
            distribuition_para = self.computeDistParameter(privacyPara, lb, ub)
            return self.sample_one_with_distribution_para(data, distribuition_para, lb, ub)
        except:
            # logger.error(f"Input: {data} can not be converted to numpy array!")
            n = data.reshape(-1).shape[0]
            lb_list = [lb for _ in range(n)]
            ub_list = [ub for _ in range(n)]
            distribuition_para = self.computeDistParameter(privacyPara, lb_list, ub_list)
            return self.sample_one_with_distribution_para(data, distribuition_para, lb_list, ub_list)
        
        
    
    def sample_many(self, data, privacyPara, lb, ub, n: int):
        '''
        Draw n sample(s) from the bounded privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - privacyPara: privacy parameter object.
            - lb: lower bound for data.
            - ub: upper bound for data.
            - n: number of samples.
        
        Return:
            - Samples
        '''
        try:
            distribuition_para = self.computeDistParameter(privacyPara, lb, ub)
        except:
            n = data.reshape(-1).shape[0]
            lb_list = [lb for _ in range(n)]
            ub_list = [ub for _ in range(n)]
            distribuition_para = self.computeDistParameter(privacyPara, lb_list, ub_list)
            
        return self.sample_many_with_distribution_para(data, distribuition_para, lb, ub, n)
        
    def sample_many_with_distribution_para(self, data, distribuition_para, lb, ub, n):
        '''
        Draw n sample(s) from the privacy mechanism with specific distribution parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - distribuition_para: distribution parameter.
            - lb: lower bound for data.
            - ub: upper bound for data.
            - n: number of samples.
        
        Return:
            - Samples
        '''
        simples = []

        for _ in range(n):
            simple = self.sample_one_with_distribution_para(data, distribuition_para, lb, ub)
            simples.append(simple)

        return simples
    

    @abstractmethod
    def sample_one_with_distribution_para(self, data, distribuition_para, lb, ub):
        '''
        Return one sample with given distribution parameter(s)
        '''
        pass

    @abstractmethod
    def computeDistParameter(self, privacyPara, lb, ub):
        '''
        Compute privacy distribution parameters with given privacy parameters
        '''
        pass