import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
import collections
from model.differential_privacy.mechanisms import boundedMech, privacyPara
from casadi import *

#from util.logging_util import init_logger
#logger = init_logger()

####################################################
##### Useful functions for bounded Laplace Mechanism
####################################################
def Cq(b, q, l, u):
    return 1 - 1/2 * (math.exp(-(q - l) / b) + math.exp(-(u - q) / b))

def fix_point_operator(b, l, u, privacyPara):
    epsilon = privacyPara.epsilon
    delta = privacyPara.delta
    sensitivity = privacyPara.sensitivity

    denominator = epsilon - math.log(Cq(b, l + sensitivity, l, u) / Cq(b, l, l, u)) - math.log(1 - delta)
    return sensitivity / denominator

###############################
##### Bounded Laplace Mechanism
###############################

class boundedLaplaceMech(boundedMech):
    def __init__(self) -> None:
        super().__init__()
    
    # overriding functions
    def sample_one_with_distribution_para(self, data, distribuition_para, lb, ub):
        temp = lb - 1
        while (temp < lb or temp > ub):
            temp = np.random.laplace(data, distribuition_para)
        
        return [temp]

    def computeDistParameter(self, privacyPara, lb, ub):
        epsilon = privacyPara.epsilon
        delta = privacyPara.delta
        sensitivity = privacyPara.sensitivity

        # initial condition
        b0 = sensitivity / (epsilon + math.log(1-delta))
        
        left = b0
        right = fix_point_operator(b0, lb, ub, privacyPara)
        intervalSize = (left + right) * 2 

        ################################
        ##### Use fix operator to find b
        ################################

        while intervalSize > right - left:
            intervalSize = right - left
            b = (left + right) / 2
            
            if fix_point_operator(b, lb, ub, privacyPara) >= b:
                left = b
            else:
                right = b    
            
        return b
    
#####################################################
##### Useful functions for bounded Gaussian Mechanism
#####################################################
def fi(x):
    """
        Return the stardard normal CDF function.
    """
    return 1/2*(1+erf(x/sqrt(2)))

def compute_num_seg(lb, ub):
    length = len(lb)
    range2count = collections.defaultdict(int)
    for i in range(length):
        range2count[(lb[i], ub[i])] += 1
    return range2count

def objective(c, range2count, standard_div):
        """
            Compute the objective function in the bounded Gaussian paper.
            We make the objective function negative since we are using the minimization.
        """
        product = 1
        
        for index, [range, count] in enumerate(range2count.items()):
            l = range[0]
            u = range[1]
            numerator = fi((u - l - c[index]) / standard_div) - fi(-c[index] / standard_div)
            denominator = (fi((u - l) / standard_div) - fi(0))
            
            product *= (numerator / denominator) ** count

        return -product

def deltaC(privacyPara, variance, range2count, bound_l2_norm):
    """
        Return Delta C
    """
    sensitivity = privacyPara.sensitivity
    num_seg = len(range2count)

    ######################################
    ##### Use Casadi to find the optimal c
    ######################################
    # decision variable
    c = SX.sym('c', num_seg, 1)

    # upper bound of decision variable c
    ubx = []

    # norm of c
    norm_c = 0

    # initial condition
    c0 = []
    for index, [range, count] in enumerate(range2count.items()):
        # initial conditions: use range middle points
        c0.append((range[1] - range[0]) / 2)

        # update norm of c
        norm_c += c[index] * count

        # update upper bound of c
        ubx.append(range[1] - range[0])
    
    # if sensitivity is large enough, return range middle points
    if sensitivity >= bound_l2_norm / 2:
        return c0
    
    c0 = np.array(c0)
    norm_c = np.sqrt(norm_c)

    # constraints
    constraint_l2_norm = sensitivity - norm_c

    # problem formulation
    prob = {}
    prob['x'] = c
    prob['f'] = objective(c,range2count, math.sqrt(variance))
    prob['g'] = constraint_l2_norm
    
    F = nlpsol('F','ipopt',prob)

    sol = F(x0 = c0,
            lbx = np.zeros((num_seg,1)),
            ubx = ubx,
            ubg = sensitivity,
            lbg = 0)
    
    c = sol['x']
    return -objective(c, range2count, math.sqrt(variance))

################################
##### Bounded Gaussian Mechanism
################################
class boundedGaussianMech(boundedMech):
    def __init__(self) -> None:
        super().__init__()

    def sample_one_with_distribution_para(self, data, distribuition_para, lb, ub):
        length = len(data)
        sample = []

        try:
            len(lb)
            lb_list = lb
            ub_list = ub
        except:
            lb_list = [lb for _ in range(length)]
            ub_list = [ub for _ in range(length)]

        for i in range(length):
            temp = lb_list[i] - 1
            while (temp < lb_list[i] or temp > ub_list[i]):
                temp = np.random.normal(data[i], math.sqrt(distribuition_para))
            sample.append(temp)

        return sample
        
    def computeDistParameter(self, privacyPara, lb, ub):
        epsilon = privacyPara.epsilon
        sensitivity = privacyPara.sensitivity

        # first check how many segments we have in lb and ub
        range2count = compute_num_seg(lb, ub)

        ############################
        #### now compute ||ub-lb||_2
        ############################
        bound_l2_norm = 0
        for range,count in range2count.items():
            bound_l2_norm += (range[1] - range[0]) * count
        bound_l2_norm = math.sqrt(bound_l2_norm)

        ##############################################################################
        ##### find optimal parameter sigma^2 (i.e., variance) using fix point operator
        ##############################################################################
        left = ((bound_l2_norm + sensitivity/2) * sensitivity) / epsilon
        DeltaC = deltaC(privacyPara = privacyPara,
                        variance = left,
                        range2count = range2count,
                        bound_l2_norm = bound_l2_norm)
        right = ((bound_l2_norm + sensitivity/2) * sensitivity) / (epsilon-np.log(DeltaC))

        intervalSize = (left + right) * 2

        while(intervalSize > right - left):
            intervalSize = right - left

            variance = (left + right) / 2
            
            DeltaC = deltaC(privacyPara = privacyPara,
                            variance = variance,
                            range2count = range2count,
                            bound_l2_norm = bound_l2_norm)
            
            fix_operator = ((bound_l2_norm + sensitivity/2) * sensitivity) / (epsilon-np.log(DeltaC))
            
            if fix_operator >= variance:
                left = variance
            else:
                right = variance

        return variance


if __name__ == '__main__':
    gaussian = boundedGaussianMech()
    privacyPara = privacyPara(epsilon=1, delta = 0.1, sensitivity=1)
    print(gaussian.sample_one([1], privacyPara, [0] ,[2]))
    print(gaussian.sample_many([1], privacyPara, [0] , [2], 10))
    print(gaussian.sample_one_with_distribution_para([1], 0.01, [0], [2]))
    print(gaussian.sample_many_with_distribution_para([16], 0.01, [0], [20], 10))