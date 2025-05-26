import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

import math
import numpy as np
import collections
from model.differential_privacy.mechanisms import boundedMech, privacyPara
from casadi import *
import torch
import cupy
from typing import Union

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
    def distribution(self, 
                     data: np.ndarray, 
                     distribuition_para: float,
                     lb: list,
                     ub: list):         
        """
        Draw one random sample
        """
        data_size = data.shape[0]
        b_size = len(lb)
        sample = []
        lb_mat = []
        ub_mat = []

        for i in range(data_size):
            for j in range(b_size):
                if data[i] > lb[j] and data[i] <= ub[j]:
                    lb_mat.append(lb[j])
                    ub_mat.append(ub[j])


        for i in range(data_size):
            temp = lb_mat[i] - 1
            while (temp < lb_mat[i] or temp > ub_mat[i]):
                temp = np.random.laplace(data[i], distribuition_para)
            sample.append(temp)
        
        return np.array(sample)

    def computeDistParameter(self, 
                             data: np.ndarray,
                             privacyPara: privacyPara):
        epsilon = privacyPara.epsilon
        delta = privacyPara.delta
        sensitivity = privacyPara.sensitivity
        lb = privacyPara.lb[0]
        ub = privacyPara.ub[0]

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
    c = SX.sym('c', num_seg, 1) # type: ignore

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
        norm_c += c[index] ** 2 * count

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

    # Do not display
    opts = {}
    opts['ipopt.print_level'] = 0
    opts['print_time'] = 0
    
    F = nlpsol('F','ipopt',prob, opts)

    sol = F(x0 = c0,
            lbx = np.zeros((num_seg,1)),
            ubx = ubx,
            ubg = sensitivity,
            lbg = 0)
    
    c = sol['x']
    return -objective(c, range2count, math.sqrt(variance))

def construct_bound_mat(data: np.ndarray,
                        lb: list,
                        ub: list):
    """
    Construct the lower bound matrix and upper bound matrix for data.
    We use binary search.

    Inputs:
        - data: sensitive data.
        - lb: list of possible lower bounds.
        - ub: list of possible upper bounds.

    Outpus:
        - lb_mat: list, a lower bound matrix which is of the same size of data.
        - up_mat: list, a upper bound matrix which is of the same size of data.
    """
    
    if (len(lb) != len(ub)):
        raise ValueError("The size of lower bound and upper should be the same!")

    # we use binary search
    n = len(lb)
    lb_mat = []
    ub_mat = []

    for d in data:
        # start binary search
        start = 0
        end = n - 1
        while (start <= end):
            mid = start + (end - start) // 2
            if d > lb[mid] and d <= ub[mid]:
                lb_mat.append(lb[mid])
                ub_mat.append(ub[mid])
                break
            elif d <= lb[mid]:
                end = mid - 1
            elif d > ub[mid]:
                start = mid + 1

    return lb_mat, ub_mat

################################
##### Bounded Gaussian Mechanism
################################
class boundedGaussianMech(boundedMech):
    def __init__(self) -> None:
        super().__init__()

    def distribution_v1(self, 
                     data: np.ndarray, 
                     distribuition_para: float,
                     lb: list,
                     ub: list):
        
        data_size = data.shape[0]
        sample = []
        lb_mat, ub_mat = construct_bound_mat(data=data,
                                             lb=lb,
                                             ub=ub)

        for i in range(data_size):
            temp = lb_mat[i] - 1
            while (temp < lb_mat[i] or temp > ub_mat[i]):
                temp = np.random.normal(data[i], math.sqrt(distribuition_para))
            sample.append(temp)

        return np.array(sample)
    
    def distribution(self, 
                          data: np.ndarray, 
                          distribuition_para: Union[float, np.ndarray],
                          lb: list,
                          ub: list):

        if torch.cuda.is_available():
            matrix = cupy
        else:
            matrix = np

        lb_mat, ub_mat = construct_bound_mat(data=data,
                                             lb=lb,
                                             ub=ub)
        
        lb_mat = np.array(lb_mat)
        ub_mat = np.array(ub_mat)
        sample = lb_mat - 1.
        while (np.any(sample < lb_mat) or np.any(sample >= ub_mat)):

            # find the data that need to be resampled
            tmp_data = data[(sample<lb_mat) | (sample>=ub_mat)]

            # draw private samples (numpy multivariate normal)
            if type(distribuition_para) == float:
                tmp_private_data = matrix.random.multivariate_normal(tmp_data, math.sqrt(distribuition_para)*np.eye(tmp_data.shape[0]))
            elif type(distribuition_para) == np.ndarray:
                dist = distribuition_para[(sample<lb_mat) | (sample>=ub_mat)]
                dist = dist[:, (sample<lb_mat) | (sample>=ub_mat)]
                tmp_private_data = matrix.random.multivariate_normal(tmp_data, np.sqrt(dist))
            else:
                raise TypeError("Wrong type of distribution parameter!")

            # put the samples
            sample[(sample<lb_mat) | (sample>=ub_mat)] = tmp_private_data.get() # type:ignore
        
        return sample
        
    def computeDistParameter(self, 
                             data: np.ndarray,
                             privacyPara: privacyPara,
                             tolerance: float = 0) -> float:
        
        epsilon = privacyPara.epsilon
        sensitivity = privacyPara.sensitivity
        delta = privacyPara.delta
        
        ###########################
        ##### construct bound array
        ###########################
        lb, ub = construct_bound_mat(data=data,
                                     lb=privacyPara.lb,
                                     ub=privacyPara.ub)


        # first check how many segments we have in lb and ub
        range2count = compute_num_seg(lb, ub)

        ############################
        #### now compute ||ub-lb||_2
        ############################
        bound_l2_norm = 0
        for r,count in range2count.items():
            bound_l2_norm += (r[1] - r[0]) ** 2 * count
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

        while(intervalSize > right - left + tolerance):
            intervalSize = right - left

            variance = (left + right) / 2
            
            DeltaC = deltaC(privacyPara = privacyPara,
                            variance = variance,
                            range2count = range2count,
                            bound_l2_norm = bound_l2_norm)
            
            if delta != 0:
                fix_operator = ((bound_l2_norm + sensitivity/2) * sensitivity) / (epsilon-np.log(DeltaC)-np.log(delta))
            else:
                fix_operator = ((bound_l2_norm + sensitivity/2) * sensitivity) / (epsilon-np.log(DeltaC))
            
            if fix_operator >= variance:
                left = variance
            else:
                right = variance

        return float(variance)                 # type: ignore


if __name__ == '__main__':
    run_laplace = 0
    run_gaussian = 1

    if run_laplace:
        laplace = boundedLaplaceMech()
        privacyPara_laplace = privacyPara(epsilon=1, delta = 0.1, sensitivity=1, lb=[0], ub=[20])
        print(laplace.sample(data = np.array([[1],[2]]), privacyPara=privacyPara_laplace, n=1)[0])
        print(laplace.sample(data = np.array([[1,2], [3,4]]), privacyPara=privacyPara_laplace, n=10)[0])
        print(laplace.sample_from_distribution(data = np.array([[1], [6]]), distribuition_para=0.01, lb=[0,10], ub=[10,20], n=1)[0])
        print(laplace.sample_from_distribution(data = np.array([[1,16], [2,7]]), distribuition_para=0.01, lb=[0,10], ub=[10,20], n=10)[0])

    if run_gaussian:
        gaussian = boundedGaussianMech()
        privacyPara_gaussian = privacyPara(epsilon=1, delta = 0.1, sensitivity=1, lb=[0,10], ub=[10,20])
        print(gaussian.sample(data = np.array([[1.],[2.]]), privacyPara=privacyPara_gaussian, n=1)[0])
        print(gaussian.sample(data = np.array([[1.,2.], [3.,4.]]), privacyPara=privacyPara_gaussian, n=10)[0])
        print(gaussian.sample_from_distribution(data = np.array([[1.], [6.]]), distribuition_para=0.01, lb=[0,10], ub=[10,20], n=1)[0])
        print(gaussian.sample_from_distribution(data = np.array([[1.,16.], [2.,7.]]), distribuition_para=0.01,lb=[0,10], ub=[10,20], n=10)[0])