# privacy mechanism class is an abstract class
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

#from util.logging_util import init_logger
#logger = init_logger()

class privacyPara:
    """The class that contains the privacy parameters."""

    def __init__(self, 
                 epsilon: float, 
                 sensitivity: float, 
                 delta: float = 0.0, 
                 lb: list = [0.0], 
                 ub: list = [0.0]):
        """
        Inputs:
            - epsilon: privacy level.
            - sensitivity: the output sensitivity.
            - delta: tolerant probability if we use approximate differential privacy.
            - lb: lower bound if we use bounded mechanism.
            - ub: upper bound if we use bounded mechanism.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
        # only for bounded mechanisms
        self.lb = lb
        self.ub = ub

        # only used for approximate dp
        self.delta = delta

class SamplesContainer():
    """
    class of a container of private distributed reproduction number samples at one time step.
    """
    def __init__(self, Samples: np.ndarray):
        """
        Inputs:
            - prvSamples: differentially private samples. Dimension: N x N x M, 
                where N is the number of communities and M is number of private samples.
        """
        self.Samples = Samples
        self.mean = np.mean(Samples, axis=2)     # Dimension: N X N
        self.std = np.std(Samples, axis=2)       # Dimension: N X N

    def __len__(self):
        return self.Samples.shape[2]
    
    def __getitem__(self, index):
        return self.Samples[:,:,index]
    
    def __iter__(self):
        for i in range(self.Samples.shape[2]):
            yield self.Samples[:,:,i]

    def add(self, sample: np.ndarray):
        """
        Add a sample to the container.
        """
        if not (self.Samples.shape[0] == sample.shape[0] and self.Samples.shape[1] == sample.shape[1]):
            raise ValueError("Wrong sample shape!")

        self.Samples = np.dstack((self.Samples, sample))

    @staticmethod
    def from_list(sample_list: list):
        """
        Convert a list of samples to a sample container.
        """
        samples = np.stack(sample_list, axis=2)
        return SamplesContainer(samples)

class privacyMech(ABC):
    '''
    Abstract class for privacy mechanisms
    '''

    def __init__(self) -> None:
        pass
    
    def sample(self, 
               data: np.ndarray, 
               privacyPara: privacyPara, 
               n: int,
               skip_zeros: bool = True) -> SamplesContainer:
        '''
        Draw n sample(s) from the privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - privacyPara: privacy parameter object.
            - n: number of samples.
            - skip_zeros: if True, the privacy mechanism will not add privacy
                to entry that is zero.
        
        Return:
            - prvSamplesContainer. A private sample container with length equals to n.
        '''

        distribuition_para = self.computeDistParameter(privacyPara)
        return self.sample_from_distribution(data, distribuition_para, n, skip_zeros)
    
    def sample_from_distribution(self, 
                                 data: np.ndarray, 
                                 distribuition_para: float, 
                                 n: int,
                                 skip_zeros: bool = True) -> SamplesContainer:
        '''
        Draw n sample(s) from the privacy mechanism with specific distribution parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - distribuition_para: distribution parameter.
            - n: number of samples.
            - skip_zeros: if True, the privacy mechanism will not add privacy
                to entry that is zero.
        
        Return:
            - prvSamplesContainer. A private sample container with length equals to n.
        '''
        samples = []

        for _ in range(n):
            
            # if we do not want to add dp to zero entries.
            if skip_zeros:
                # find all nonzero entries
                mask = data != 0
                values = data[mask]

                # feed to dp mechanism
                s = self.distribution(data=values, distribuition_para=distribuition_para)

                # generate a private matrix
                matrix_w_dp = np.zeros((data.shape[0], data.shape[1]))
                matrix_w_dp[mask] = s

            # add dp to zero entries.
            else:
                s = self.distribution(data = data.reshape(-1), distribuition_para=distribuition_para)
                matrix_w_dp = s.reshape((data.shape[0], data.shape[1]))

            samples.append(matrix_w_dp)

        return SamplesContainer.from_list(samples)
    
    @abstractmethod
    def distribution(self, 
                     data: np.ndarray,
                     distribuition_para: float) -> np.ndarray:
        '''
        Return one sample with given distribution parameter(s)
        '''
        pass

    @abstractmethod
    def computeDistParameter(self, 
                             privacyPara: privacyPara) -> float:
        '''
        Compute privacy distribution parameters with given privacy parameters
        '''
        pass

class boundedMech(ABC):
    '''
    Abstract class for privacy mechanisms
    '''

    def __init__(self) -> None:
        pass
    
    def sample(self, 
               data: np.ndarray, 
               privacyPara: privacyPara, 
               n: int,
               skip_zeros: bool = True) -> SamplesContainer:
        '''
        Draw n sample(s) from the privacy mechanism with specific privacy parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - privacyPara: privacy parameter object.
            - n: number of samples.
            - skip_zeros: if True, the privacy mechanism will not add privacy
                to entry that is zero.
        
        Return:
            - prvSamplesContainer. A private sample container with length equals to n.
        '''

        if skip_zeros:
            # find all nonzero entries
            mask = data != 0
            values = data[mask]
        else:
            values = data.reshape(-1)

        distribuition_para = self.computeDistParameter(values, privacyPara)
        return self.sample_from_distribution(data=data, 
                                             distribuition_para=distribuition_para, 
                                             lb=privacyPara.lb,
                                             ub=privacyPara.ub,
                                             n=n, 
                                             skip_zeros=skip_zeros)
    
    def sample_from_distribution(self, 
                                 data: np.ndarray, 
                                 distribuition_para: Union[float, np.ndarray], 
                                 lb: list,
                                 ub: list,
                                 n: int,
                                 skip_zeros: bool = True) -> SamplesContainer:
        '''
        Draw n sample(s) from the privacy mechanism with specific distribution parameters.

        Input:
            - data: sensitive data. Support type: list, numpy array.
            - distribuition_para: distribution parameter.
            - n: number of samples.
            - skip_zeros: if True, the privacy mechanism will not add privacy
                to entry that is zero.
        
        Return:
            - prvSamplesContainer. A private sample container with length equals to n.
        '''
        samples = []

        for _ in range(n):
            
            # if we do not want to add dp to zero entries.
            if skip_zeros:
                # find all nonzero entries
                mask = data != 0
                values = data[mask]

                # feed to dp mechanism
                s = self.distribution(values, distribuition_para, lb=lb, ub=ub)

                # generate a private matrix
                matrix_w_dp = np.zeros((data.shape[0], data.shape[1]))
                matrix_w_dp[mask] = s

            # add dp to zero entries.
            else:
                s = self.distribution(data.reshape(-1), distribuition_para, lb=lb, ub=ub)
                matrix_w_dp = s.reshape((data.shape[0], data.shape[1]))

            samples.append(matrix_w_dp)

        return SamplesContainer.from_list(samples)
    
    @abstractmethod
    def distribution(self, 
                     data: np.ndarray,
                     distribuition_para: Union[float, np.ndarray],
                     lb: list,
                     ub: list) -> np.ndarray:
        '''
        Return one sample with given distribution parameter(s)
        '''
        pass

    @abstractmethod
    def computeDistParameter(self, 
                             data: np.ndarray,
                             privacyPara: privacyPara) -> float:
        '''
        Compute privacy distribution parameters with given privacy parameters
        '''
        pass




