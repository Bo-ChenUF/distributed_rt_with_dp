import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
from model.differential_privacy.mechanisms import privacyMech, privacyPara
#from util.logging_util import init_logger
#logger = init_logger()

class gaussianMech(privacyMech):
    '''
    Gaussian mechanism
    '''
    def __init__(self) -> None:
        super().__init__()

    # overriding functions
    def sample_one_with_distribution_para(self, data, distribuition_para):
        '''
        Return a random sample for the given sensitive data. If data is multidimensional, return a 
        multidimensional sample, where each dimension of the sample is drawn using the same distribution
        parameter sigmasq (i.e., variance).
        '''
        length = len(data)
        #logger.info(f"Sensitive Data Length:{length}")
        simple = []

        # timer
        #logger.info('Start drawing samples...')

        for i in range(length):
            simple.append(np.random.normal(data[i], math.sqrt(distribuition_para)))

        #logger.info('End drawing samples...')
        
        return [simple]
    
    def computeDistParameter(self, privacyPara):
        '''
        Compute Gaussian distribution parameter sigmasq (variance)
        '''
        variance = 2 * math.log(1.25/privacyPara.delta) * (privacyPara.sensitivity**2) / (privacyPara.epsilon**2)
        #logger.info(f"Gaussian Mechanism variance={variance}")
        return variance
    

class laplaceMech(privacyMech):
    def __init__(self) -> None:
        super().__init__()

    # overriding functions
    def sample_one_with_distribution_para(self, data, distribuition_para):
        '''
        Return a random sample for the given sensitive data. For Laplace mechanism, data can only be 1-dimensional
        '''
        return [np.random.laplace(data, distribuition_para)]

    def computeDistParameter(self, privacyPara):
        '''
        Compute Laplace distribution parameter b.
        '''
        b = privacyPara.sensitivity / privacyPara.epsilon
        #logger.info(f"Laplace mechanism parameter b={b}")
        return b
    

if __name__ == '__main__':

    gaussian = gaussianMech()
    privacyPara = privacyPara(epsilon=1, delta = 0.1, sensitivity=1)
    print(gaussian.sample_one([1], privacyPara))
    print(gaussian.sample_many([1], privacyPara, 10))
    print(gaussian.sample_one_with_distribution_para([1], 0.01))
    print(gaussian.sample_many_with_distribution_para([16], 0.01, 10))