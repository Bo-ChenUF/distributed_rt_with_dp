import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.notebook_util import spectral_radius
from model.differential_privacy.mechanisms import privacyPara
from model.differential_privacy.bounded_mechanisms import boundedGaussianMech

import numpy as np

class distributed_repro_num():
    """
        class of the distributed reproduction number.
    """
    def __init__(self, matrix) -> None:
        self.matrix = matrix

    def get_overall_repro_number(self) -> float:
        """ Return the overall reproduction number for all communities"""
        
        return spectral_radius(self.matrix)
        #return float(np.sum(self.matrix) / self.matrix.shape[0])
    
class central_authority(distributed_repro_num):
    """ class of central authority which can add noise to distributed repro numbers"""
    def __init__(self, matrix) -> None:
        super().__init__(matrix)

    def add_dp(self, 
               epsilon: float, 
               sensitivity: float, 
               lb: float,
               ub: float,
               num_samples: int, 
               type="bounded_gaussian"):
        """
            Add differential privacy to the reproduction number matrix

            Input:
                - epsilon: privacy parameter
                - sensitivity: privacy parameter
                - lb: feasible range lower bound
                - ub: feasible range upper bound
                - num_samples: number of private samples
                - type: type of noise for privacy

            Output:
                - list of distributed repro objects
        """

        # initialize privacy mechanism object
        if type == 'bounded_gaussian':
            mechanism = boundedGaussianMech()
        else:
            #TODO: we may add other privacy mechanisms if necessnary.
            raise ValueError("For current version of simulation, only bounded gaussian mechanism is allowed!")
        
        # unpack the matrix
        matrix = self.matrix

        # initialize privacypara object
        para = privacyPara(epsilon=epsilon,
                           sensitivity=sensitivity)
        
        # Feed all nonzero entries to dp mechanism
        mask = matrix != 0
        values = matrix[mask]

        # Get private samples
        private_samples = mechanism.sample_many(data=values, 
                                                privacyPara=para, 
                                                lb=lb, 
                                                ub=ub, 
                                                n=num_samples)

        # initialize result list
        res = []
        matrix_w_dp = np.zeros((matrix.shape[0], matrix.shape[1]))
        for s in private_samples:
            matrix_w_dp[mask] = np.array(s)
            res.append(distributed_repro_num(matrix_w_dp))
        
        return res

if __name__ == "__main__":
    # For debug
    # Example reproduction matrix
    example_matrix = np.array([[1,2,0],
                               [2,3,1],
                               [3,2,1]])
    matrix = central_authority(example_matrix)

    # simulation parameters
    epsilon = 1
    sensitivity = 0.001
    lb = 0
    ub = 4
    num_samples = 10

    # Get private samples
    private_samples = matrix.add_dp(epsilon=epsilon,
                                    sensitivity=sensitivity,
                                    lb=lb,
                                    ub=ub,
                                    num_samples=num_samples)
    
    print(private_samples[0].matrix)