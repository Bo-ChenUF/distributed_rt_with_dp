import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
from model.differential_privacy.mechanisms import privacyMech, privacyPara, SamplesContainer
#from util.logging_util import init_logger
#logger = init_logger()

class gaussianMech(privacyMech):
    '''
    Gaussian mechanism
    '''
    def __init__(self) -> None:
        super().__init__()

    # overriding functions
    def distribution(self, 
                     data: np.ndarray, 
                     distribuition_para: float) -> np.ndarray:
        '''
        Return a random sample for the given sensitive data. If data is multidimensional, return a 
        multidimensional sample, where each dimension of the sample is drawn using the same distribution
        parameter sigmasq (i.e., variance).

        Inputs:
            - data: sensitive data.
            - distribution_para: the distribution parameter, in this case, the variance sigma_sq.
            - skip_zeros: if True, the privacy mechanism will not add privacy
                to entry that is zero.
        
        Output:
            - prvSample: a private sample.
        '''
        length = data.shape[0]
        #logger.info(f"Sensitive Data Length:{length}")
        samples = []

        # timer
        #logger.info('Start drawing samples...')

        for i in range(length):
            samples.append(np.random.normal(data[i], math.sqrt(distribuition_para)))

        #logger.info('End drawing samples...')
        
        return np.array(samples)
    
    def computeDistParameter(self, 
                             privacyPara: privacyPara) -> float:
        '''
        Compute Gaussian distribution parameter sigmasq (variance)
        '''
        variance = 2 * math.log(1.25/privacyPara.delta) * (privacyPara.sensitivity**2) / (privacyPara.epsilon**2)
        return variance
    

class laplaceMech(privacyMech):
    def __init__(self) -> None:
        super().__init__()

    # overriding functions
    def distribution(self, data, distribuition_para):
        '''
        Return a random sample for the given sensitive data.
        '''
        return np.random.laplace(data, distribuition_para)

    def computeDistParameter(self, 
                             privacyPara: privacyPara):          # type: ignore
        '''
        Compute Laplace distribution parameter b.
        '''
        b = privacyPara.sensitivity / privacyPara.epsilon
        return b
    

if __name__ == '__main__':
    run_laplace = 0
    run_gaussian = 1

    if run_laplace:
        laplace = laplaceMech()
        privacyPara_laplace = privacyPara(epsilon=1, delta = 0.1, sensitivity=1)
        print(laplace.sample(data = np.array([[1],[2]]), privacyPara=privacyPara_laplace, n=1)[0])
        print(laplace.sample(data = np.array([[1,2], [3,4]]), privacyPara=privacyPara_laplace, n=10)[0])
        print(laplace.sample_from_distribution(data = np.array([[1], [6]]), distribuition_para=0.01, n=1)[0])
        print(laplace.sample_from_distribution(data = np.array([[1,16], [2,7]]), distribuition_para=0.01, n=10)[0])

    if run_gaussian:
        gaussian = gaussianMech()
        privacyPara_gaussian = privacyPara(epsilon=1, delta = 0.1, sensitivity=1)
        print(gaussian.sample(data = np.array([[1],[2]]), privacyPara=privacyPara_gaussian, n=1)[0])
        print(gaussian.sample(data = np.array([[1,2], [3,4]]), privacyPara=privacyPara_gaussian, n=10)[0])
        print(gaussian.sample_from_distribution(data = np.array([[1], [6]]), distribuition_para=0.01, n=1)[0])
        print(gaussian.sample_from_distribution(data = np.array([[1,16], [2,7]]), distribuition_para=0.01, n=10)[0])