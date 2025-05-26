import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.differential_privacy.mechanisms import privacyPara, SamplesContainer
from model.differential_privacy.bounded_mechanisms import boundedGaussianMech
from abc import ABC, abstractmethod

import numpy as np

def spectral_radius(matrix):
    """
        Compute spectral radius of the input matrix.

        Input:
            -matrix: numpy array
        
        Output:
            spectral radius
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return max(abs(eigenvalues))

def compute_eff_rep_num_mat(sus_portion: np.ndarray, 
                            inf_portion: np.ndarray,
                            basic_rep_mat: np.ndarray,
                            population: np.ndarray) -> np.ndarray:
    """ 
    Return the effective reproduction number matrix.

    Inputs:
        - basic_rep_mat: numpy.ndarray. The basic reproduction number. N by N, where N -> number of clusters.
        - sus_portion: numpy.ndarray. Suspected portion. N by 1, where N -> number of clusters.
        - inf_portion: numpy.ndarray. Infected portion. N by 1, where N -> number of clusters.
        - population: numpy.ndarray. Population of clusters. N by 1, where N -> number of clusters.

    Output:
        - eff_rep_num_mat: numpy.ndarray. The effective reproduction number. N by N, where N -> number of clusters.
    """

    # Sanity check
    if (sus_portion.shape[0] != basic_rep_mat.shape[0] or inf_portion.shape[0] != basic_rep_mat.shape[0] or 
        population.shape[0] != basic_rep_mat.shape[0] or basic_rep_mat.shape[0] != basic_rep_mat.shape[1]):
        raise ValueError("Inconsistent dimensionality! Please check dimensions of inputs!")

    N = basic_rep_mat.shape[0]
    eff_rep_num_mat = np.zeros((N,N))

    # compute effective reproduction number
    for i in range(N):
        for j in range(N):

            # check infected portion
            if (inf_portion[i] == 0 and inf_portion[j] == 0):
                # if both communities have 0 infected portion, then reproduction number is 0
                #eff_rep_num_mat[i][j] = 0
                eff_rep_num_mat[i][j] = sus_portion[i] * basic_rep_mat[i][j]
            elif (inf_portion[i] == 0):
                # if the income community have 0 infected portion, 
                # the we assume one person is infected in that community to avoid computational problems.
                eff_rep_num_mat[i][j] = sus_portion[i] * inf_portion[j] * basic_rep_mat[i][j] * (population[i])
            else:
                eff_rep_num_mat[i][j] = sus_portion[i] * inf_portion[j] * basic_rep_mat[i][j] * (1.0/inf_portion[i])

    return eff_rep_num_mat

class distRepNum():
    """
        class of the distributed effective reproduction number.
    """
    def __init__(self, eff_rep_num_mat: np.ndarray) -> None:
        self.eff_rep_num_mat = eff_rep_num_mat      # effective reproduction number matrix, dimension: N by N,  N -> number of cluster

    def __getitem__(self, pos):
        r, c = pos
        return self.eff_rep_num_mat[r,c]
    
    def add_dp_v1(self, 
               privacyPara: privacyPara,
               num_samples: int,
               type: str = 'bounded_gaussian') -> SamplesContainer:
        """
        Add differential privacy to the reproduction number matrix

        Input:
            - privacyPara: the collection of privacy parameters
            - num_samples: int, number of private samples.
            - type: type of noise for privacy

        Output:
            - SamplesContainer which contains all the private samples.
        """

        # initialize privacy mechanism object
        if type == 'bounded_gaussian':
            mechanism = boundedGaussianMech()
        else:
            #TODO: we may add other privacy mechanisms if necessnary.
            raise ValueError("Unrecognizable type of privacy mechanism!")

        # unpack the matrix
        matrix = self.eff_rep_num_mat
        size = matrix.shape[0]

        samples = []
        for i in range(matrix.shape[0]):
            print(f"Current Community id: {i}")
            # Get private samples
            samples.append(mechanism.sample(data=matrix[i].reshape((size,1)),
                                            privacyPara=privacyPara,
                                            n=num_samples))
        
        res = []
        for i in range(num_samples):
            s = np.zeros((size, size))
            for j in range(len(samples)):
                s[j,:] = samples[j][i].reshape(-1)

            res.append(s)

        return SamplesContainer.from_list(res)

    def add_dp(self, 
                    privacyPara: privacyPara,
                    num_samples: int,
                    type: str = 'bounded_gaussian') -> SamplesContainer:
        """
        Add differential privacy to the reproduction number matrix

        Input:
            - privacyPara: the collection of privacy parameters
            - num_samples: int, number of private samples.
            - type: type of noise for privacy

        Output:
            - SamplesContainer which contains all the private samples.
        """
        # initialize privacy mechanism object
        if type == 'bounded_gaussian':
            mechanism = boundedGaussianMech()
        else:
            #TODO: we may add other privacy mechanisms if necessnary.
            raise ValueError("Unrecognizable type of privacy mechanism!")

        # unpack the matrix
        matrix = self.eff_rep_num_mat
        size = matrix.shape[1]

        # we first compute the distribution parameter
        # in the privacy mechanism for each community
        dist_list = []
        for i in range(matrix.shape[0]):
            print(f"Current Community id: {i}")

            # Get the distributed rep number for this community
            data=matrix[i].reshape((size,1))

            # Find all non zero entries
            mask = data != 0
            values = data[mask]

            # Get private samples
            dist_list.extend([mechanism.computeDistParameter(data=values,
                                                             privacyPara=privacyPara)] * values.shape[0])
        
        dist = np.diag(dist_list)

        ####################
        ##### Start sampling
        ####################
        res = mechanism.sample_from_distribution(data=matrix,
                                                 distribuition_para=dist, 
                                                 lb=privacyPara.lb,
                                                 ub=privacyPara.ub,
                                                 n=num_samples)

        return res



    @staticmethod
    def from_epi_anl(basic_rep_mat: np.ndarray,
                     sus_portion: np.ndarray,
                     inf_portion: np.ndarray,
                     population: np.ndarray,
                     clip_value : int = 20):
        """
        Compute the effective reproduction number from basic reproduction number,
        suspected portion, and infected portion.

        Inputs:
            - basic_rep_mat: numpy.ndarray. The basic reproduction number. N by N, where N -> number of clusters.
            - sus_portion: numpy.ndarray. Suspected portion. N by 1, where N -> number of clusters.
            - inf_portion: numpy.ndarray. Infected portion. N by 1, where N -> number of clusters.
            - population: numpy.ndarray. Population of clusters. N by 1, where N -> number of clusters.
            - clip_value: int. A cap for reproduction number.

        Output:
            - An instance of distRepNum class.
        """
        eff_rep_num_mat = compute_eff_rep_num_mat(sus_portion, inf_portion, basic_rep_mat, population)
        eff_rep_num_mat = np.clip(eff_rep_num_mat, 0, clip_value)
        return distRepNum(eff_rep_num_mat)

class Collection(ABC):
    """
    An abstract class for data/sample collection.
    """
    def __init__(self):
        """
        self.samples: List of SamplesContainer of size D (number of days), each sample container stores a matrix of size N*N*M
                        where N is the number of communities, and M is the number of (private) samples.
        """
        self.samples = []

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def __iter__(self):
        for s in self.samples:
            yield s

class sampleCollection(Collection):
    """
    class of a collection of samples collected for multiple days
    """
    def __init__(self):
        """
        self.samples: List of SamplesContainer of size D (number of days), each sample container stores a matrix of size N*N*M
                        where N is the number of communities, and M is the number of (private) samples.
        self.samples_mean: List of numpy array of size D (number of days), mean of (private) samples.
        self.samples_std: List of numpy array of size D (number of days), standard diviation of (private) samples.
        """
        super().__init__()        
        self.samples_mean = []
        self.samples_std = []

    def get_mean(self):
        return self.samples_mean
    
    def get_std(self):
        return self.samples_std
    
    def add(self, samples: SamplesContainer):
        """
        Add a sample container to this sample collector
        """
        self.samples.append(samples)
        self.samples_mean.append(samples.mean)
        self.samples_std.append(samples.std)

class dataCollection(Collection):
    """
    class of a collection of distributed reproduction number matrix.
    """
    def __init__(self):
        super().__init__()

    def add(self, data: distRepNum):
        """
        Add a sample container to this sample collector
        """
        self.samples.append(data)

    def add_dp(self,
               privacyPara: privacyPara,
               num_samples: int,
               type: str = 'bounded_gaussian',
               start_index: int = 0,
               end_index = None) -> sampleCollection:        
        """
        Add differential privacy to the distributed reproduction number at a given date range [start_index, end_index].

        Inputs:
            - privacyPara: the collection of privacy parameters
            - num_samples: int, number of private samples.
            - type: the type of privacy mechanism.
            - start_index: int, starting index
            - end_index: int, ending index 

        Output:
            - samplesContainer: a collection of samples collected for these days.
        """
        
        if end_index:
            # index checking
            if end_index+1 > len(self.samples):
                raise ValueError("End index out of bound!")

            end_point = end_index + 1
        else:
            end_point = len(self.samples)

        collection = sampleCollection()
        for i in range(start_index, end_point):
            print(f"Current day: {i}")
            collection.add(self.samples[i].add_dp(privacyPara, num_samples, type))

        return collection



if __name__ == "__main__":
    # For debug
    # Example reproduction matrix
    example_matrix = np.array([[1,2,0],
                               [2,3,1],
                               [3,2,1]])
    matrix = distRepNum(example_matrix)

    # simulation parameters
    prvPara = privacyPara(epsilon = 1, sensitivity = 0.001, lb = [0], ub = [4])
    num_samples = 100

    # Get private samples
    private_samples = matrix.add_dp(privacyPara=prvPara,
                                    num_samples=num_samples)
    
    print(private_samples[0])
