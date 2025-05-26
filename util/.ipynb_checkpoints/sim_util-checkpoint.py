import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.central_authority import distRepNum, dataCollection

def ensemble_eff_rep_old(M : int,
         population: np.ndarray,
         firstClu2SecondClu: dict, 
         simpId2cluId: dict,
         inf_portion: np.ndarray, 
         matrix: np.ndarray,
         n_cluster_list: list,
         clip_value: int = 20) -> np.ndarray:
    """
    Get the distributed effective reproduction number after the second clustering

    Input:
        - M: int, number of clusters we are interested in.
        - population: numpy.ndarray. Population of clusters. N by 1, where N -> number of clusters.
        - firstClu2SecondClu: dictionary, the map from the cluster_id after the first clustering 
            to the cluster_id after the second clustering. Length: N -> the number of clusters after first clustering
        - sus_portion: suspected portion after the first clustering. Dimension: N by D.
            M -> the number of clusters after first clustering, D -> number of dates.
        - eff_rep_num_mat_list: a list of effective repreductive number matrices with length D
        - n_cluster_list: list of clusters.
        - clip_value: int. A cap for reproduction number.

    Output:
        - res: a list of effective repreductive number matrices after second clustering with length D.
    """
    N = inf_portion.shape[0] # the number of clusters after first clustering
    
    eff_mat = np.zeros((M,M))
    eff_mat_prev = matrix
    denomiator = np.zeros(M)
    
    for x in range(N):
        for y in range(N):
            tmp_x = firstClu2SecondClu[simpId2cluId[x]]
            tmp_y = firstClu2SecondClu[simpId2cluId[y]]

            # map_x to eff_mat index
            for i, node in enumerate(n_cluster_list):
                if tmp_x == node:
                    map_x = i
                if tmp_y == node:
                    map_y = i
            
            if (inf_portion[x] == 0):
                # if the infected portion is 0, we assume at lease one person is
                # infected to prevent computational problem
                eff_mat[map_x][map_y] += (1.0 / population[x]) * eff_mat_prev[x][y]
            else:
                eff_mat[map_x][map_y] += inf_portion[x] * eff_mat_prev[x][y]

        if (inf_portion[x] == 0):
            # Similarly, we assume at lease one person is
            # infected to prevent computational problem
            denomiator[map_x] += (1.0 / population[x])
        else:
            denomiator[map_x] += inf_portion[x]

    for i in range(M):
        eff_mat[i, :] = eff_mat[i, :] / denomiator[i]

    eff_mat = np.clip(eff_mat, 0, clip_value)

    return eff_mat

def ensemble_eff_rep(M : int,
         population: np.ndarray,
         firstClu2SecondClu: dict, 
         simpId2cluId: dict,
         inf_portion: np.ndarray, 
         matrix: np.ndarray,
         n_cluster_list: list,
         clip_value: int = 20) -> np.ndarray:
    """
    Get the distributed effective reproduction number after the second clustering

    Input:
        - M: int, number of clusters we are interested in.
        - population: numpy.ndarray. Population of clusters. N by 1, where N -> number of clusters.
        - firstClu2SecondClu: dictionary, the map from the cluster_id after the first clustering 
            to the cluster_id after the second clustering. Length: N -> the number of clusters after first clustering
        - sus_portion: suspected portion after the first clustering. Dimension: N by D.
            M -> the number of clusters after first clustering, D -> number of dates.
        - eff_rep_num_mat_list: a list of effective repreductive number matrices with length D
        - n_cluster_list: list of clusters.
        - clip_value: int. A cap for reproduction number.

    Output:
        - res: a list of effective repreductive number matrices after second clustering with length D.
    """
    N = inf_portion.shape[0] # the number of clusters after first clustering
    
    eff_mat = np.zeros((M,M))
    eff_mat_prev = matrix
    denomiator = np.zeros(M)
    
    for x in range(N):
        for y in range(M):
            tmp_x = firstClu2SecondClu[simpId2cluId[x]]

            # map_x to eff_mat index
            for i, node in enumerate(n_cluster_list):
                if tmp_x == node:
                    map_x = i
            
            if (inf_portion[x] == 0):
                # if the infected portion is 0, we assume at lease one person is
                # infected to prevent computational problem
                eff_mat[map_x][y] += (1.0 / population[x]) * eff_mat_prev[x][y]
            else:
                eff_mat[map_x][y] += inf_portion[x] * eff_mat_prev[x][y]

        if (inf_portion[x] == 0):
            # Similarly, we assume at lease one person is
            # infected to prevent computational problem
            denomiator[map_x] += (1.0 / population[x])
        else:
            denomiator[map_x] += inf_portion[x]

    for i in range(M):
        eff_mat[i, :] = eff_mat[i, :] / denomiator[i]

    eff_mat = np.clip(eff_mat, 0, clip_value)

    return eff_mat