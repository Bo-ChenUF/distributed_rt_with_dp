import numpy as np
import pandas as pd
import datetime
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.central_authority import distRepNum, spectral_radius, dataCollection

def binary_search_beta(start, end, target, Rec_matrix_inv, contact_matrix, diagonal: bool = False):
    """
        Search for beta such that the resulting reproduction number matches the 
        given reproduction number.

        Input:
            - start: starting point
            - end: ending point
            - target: target reproduction number
            - Rec_matrix_inv: inverse recovery matrix
            - contact matrix: contact matrix
            - diagonal: beta only multiplied on diagonal values
        
        Output:
            - beta: value of beta
    """

    # corner case:
    if diagonal:
        temp = contact_matrix.copy()
        np.fill_diagonal(temp, np.diag(temp)*end)
        if spectral_radius(Rec_matrix_inv @ temp) < target:
            raise ValueError("The target value is not in the given range! Please raise the value of 'end'")
    else:
        if spectral_radius(end * Rec_matrix_inv @ contact_matrix) < target or spectral_radius(start * Rec_matrix_inv @ contact_matrix) > target:
            raise ValueError("The target value is not in the given range! Please raise the value of 'end'")

    tolerance = 0.1
    recursive_max = 100
    count = 0

    while start < end or count <= recursive_max:
        # middle point
        mid = start + (end-start) / 2

        if diagonal:
            temp = contact_matrix.copy()
            np.fill_diagonal(temp, np.diag(temp)*mid)
            product_matrix = Rec_matrix_inv @ temp
        else:
            product_matrix =  mid * Rec_matrix_inv @ contact_matrix

        # Compute spectral radius
        radius = spectral_radius(product_matrix)

        # check if the target value is reached.
        if (abs(target - radius) <= tolerance): 
            break

        if radius < target:
            start = mid # Increase beta
        else:
            end = mid  # Decrease beta

        count += 1
        print(f"current count is {count}")
        print(f"current difference: {abs(target - radius)}")

    # The following lines will be triggered if the max number 
    # of recursions reached.
    if (count > recursive_max):
        print("Max number of recursions reached!")

    return mid

def convert_col_name_to_datetime(df: pd.DataFrame) -> pd.DataFrame: 

    """
    Convert the column name to datetime format: %m/%d/%y

    Input:
        - df: pandas dataframe

    Output:
        - df: pandas dataframe with columns converted.    
    """
    rename_columns = {}

    for col in df.columns:
        
        # if the column can be convert to datetime
        try:
            t = col.to_pydatetime().date().strftime("%m/%d/%y") # type: ignore
            rename_columns[col] = t

        # else, do nothing
        except:
            continue

    # rename columns
    df = df.rename(columns=rename_columns)
    #print(rename_columns)
    return df

def moving_average(df: pd.DataFrame,  
                   smooth_day: int = 3) -> pd.DataFrame:
    """
    Given the rows as cluster id and columns as the date, smooth the data along the time range.
    here we use moving average.

    Inputs:
        - df: dataframe. Rows are cluster ids and columns are the dates.
        - smooth_day: int, the window of the moving average equals to 2 * smooth_day + 1.
    
    """
    D = len(df.columns)
    res = pd.DataFrame()

    # smooth the resulting dataframe
    for index, col in enumerate(df.columns):
        if (index >= smooth_day and index <= D - smooth_day):
            print(df.iloc[:, index - smooth_day : (index + smooth_day + 1)].sum(axis=1))
            res[col] = df.iloc[:, index - smooth_day : (index + smooth_day + 1)].sum(axis=1) / (2*smooth_day+1)
            
        else:
            res[col] = df[col]

    return res

def get_infected_population(df_confirmed_case: pd.DataFrame, 
                            recovery_day: int = 14) -> pd.DataFrame:
    """
    Given the daily confirmed cases, find the infected populations at each day.

    Input:
        - df_confirmed_case: pandas dataframe of daily confirmed cases for each cluster
            Index are clusters id, columns are dates.
        - recovery_day: The days required for a person to recover from a disease.

    Output:
        - res: pandas dataframe for infected population at each day for each cluster.
    """
    inf_case = pd.DataFrame()
    D = len(df_confirmed_case.columns)
    
    for index, col in enumerate(df_confirmed_case.columns):
        if index < recovery_day:
            # for the first {recovery_day} days, 
            # the accumulative infected number is the infected population at each day
            inf_case[col] = df_confirmed_case[col]
        else:
            # else, the infected population at each day is the current accumulative infected number
            # subtract the accumulative infected number at {recovery_day} days before.
            inf_case[col] = df_confirmed_case[col].values - df_confirmed_case.iloc[:, index - recovery_day]

    res = convert_col_name_to_datetime(inf_case)
    return res
    
def get_infected_matrix(inf_matrix: np.ndarray, populations: np.ndarray, r_min: float, r_max:float) -> np.ndarray:
    """
    Given the infected matrix from the work of Treut et.al., substitute their self-loop weights 
    with average weights among in-nodes scaled by a population-based parameter.

    Inputs:
        - inf_matrix: infected matrix from the work of Treut et.al., dimensions: 1023 X 1023
        - populations: population of each community, dimensions: 1023 X 1.
        - r_min: scale parameter, min value of the range mapping to.
        - r_max: scale parameter, max value of the range mapping to.


    Output:
        - res: the new infected matrix.
    """

    ##############################################################
    ##### If we want to use average, uncomment the following lines
    ##############################################################
    # compute the sum of each row.
    # row_sum = np.sum(inf_matrix, axis=1)

    # compute number of nonzeros at each row.
    # in_degrees = np.count_nonzero(inf_matrix, axis=1)

    ###########################
    ##### If we want to use max
    ###########################
    res = np.copy(inf_matrix)
    np.fill_diagonal(res, 0)

    # alternatively, we compute the max of each row
    row_max = np.max(res, axis=1)

    # create the map: [p_min, p_max] -> [0, scale]
    p_min = np.min(populations)
    p_max = np.max(populations)

    def mapping(value):
        """map: [p_min, p_max] -> [r_min, r_max]"""
        return (value-p_min) / (p_max-p_min) * (r_max-r_min) + r_min

    diag = row_max * mapping(populations)
    np.fill_diagonal(res, diag)
    '''
    # put value to the matrix
    for i in range(res.shape[0]):

        # for average
        # res[i][i] = row_sum[i] / in_degrees[i] * mapping(populations[i])

        # for max
        res[i][i] = row_max[i] * mapping(populations[i])
    '''

    return res

def get_infected_matrix_updated(inf_matrix: np.ndarray, rec_matrix_inv:np.ndarray, target_r0: int, end: int):
    """
    Get the infected matrix that has the global r0 equals to the target value.

    Inputs:
        - inf_matrix: infected matrix from the work of Treut et.al., dimensions: 1023 X 1023
        - rec_matrix: recovery matrix, diagonal, dimensions: 1023 X 1023
        - target_r0:  the target value of r0
        - end: the end value of the binary search

    Output:
        - res: the new infected matrix.
    """
    res = inf_matrix.copy()
    np.fill_diagonal(res, binary_search_beta(start = 0.00001, 
                                                            end = end, 
                                                            target = target_r0, 
                                                            Rec_matrix_inv = rec_matrix_inv, 
                                                            contact_matrix = inf_matrix, 
                                                            diagonal = True) * np.diag(inf_matrix))

    return res


def remove_zero_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the community with 0 population.

    Input:
        - df: pandas dataframe. 

    Output:
        - df_new: pandas dataframe with nonzero population 
    """

    df_new = df[df["population"] != 0].reset_index(drop=True)

    return df_new

def get_epi_analysis(df_clusters: pd.DataFrame, 
                     df_confirmed_case: pd.DataFrame, 
                     recovery_day: int,
                     smooth: bool = True,
                     smooth_day: int = 3):
    """
    Given the population information of each cluster and the daily confirmed cases
    Return the suspect population/portion and infected population/portion 

    Input:
        - df_clusters: dataframe of basic info of ordered clusters. Index 
            are clusters id, columns are ['leaves', 'X', 'Y', 'population'].
        - df_confirmed_case: dataframe of daily confirmed cases. Index 
            are clusters id, columns are dates.
        - recovery_day: days required for recovery
        - smooth: boolean, if True, the final result will be smoothed. if smooth, for each date, the  
            the infected number will be a moving average.
        - smooth_day: int, the window of the moving average equals to 2 * smooth_day + 1.

    Output:
        - df_sus_population: dataframe of suspected population for each cluster.
        - df_sus_portion: dataframe of suspected portion for each cluster.
        - df_inf_population: dataframe of infected population for each cluster.
        - df_inf_portion: dataframe of infected portion for each cluster.
    """
    ##############################
    ##### Get suspected population
    ##############################
    df_sus_population = df_clusters[["population"]].values - df_confirmed_case[df_confirmed_case.columns]

    #######################
    ##### suspected portion
    #######################
    df_sus_portion = df_sus_population[df_sus_population.columns] / df_clusters[["population"]].values # type: ignore
    df_sus_portion = convert_col_name_to_datetime(df_sus_portion) # type: ignore
    
    # smooth the resulting dataframe
    if smooth:
        df_sus_portion = moving_average(df=df_sus_portion, smooth_day=smooth_day)
    
    #########################
    ##### construct dataframe
    #########################
    df_inf_population = get_infected_population(df_confirmed_case, recovery_day)

    ######################
    ##### infected portion
    ######################
    df_inf_portion = df_inf_population[df_inf_population.columns] / df_clusters[["population"]].values
    df_inf_portion = convert_col_name_to_datetime(df_inf_portion) # type: ignore

    '''
    # smooth the resulting dataframe
    if smooth:
        df_inf_portion = moving_average(df=df_inf_portion, smooth_day=smooth_day)
    '''

    return df_sus_population, df_sus_portion, df_inf_population, df_inf_portion

def compute_eff_rep_num(sus_portion: np.ndarray, 
                        inf_portion: np.ndarray, 
                        basic_rep_num_mat: np.ndarray,
                        population: np.ndarray) -> dataCollection:
    """
    Compute the effective reproduction number matrix for each date.

    Input:
        - sus_portion: numpy array of suspected portion for each cluster.
            Dimension: N by D, where N -> number of cluster, D -> number of days
        - inf_portion: numpy array of infected portion for each cluster.
            Dimension: N by D, where N -> number of cluster, D -> number of days
        - basic_rep_num_mat: numpy array of basic reproduction number matrix.
            Dimension: N by N, where N -> number of cluster
        - population: numpy.ndarray. Population of clusters. 
            Dimension: N by 1, where N -> number of clusters.


    Output:
        - eff_rep_num_list: List of effective reproduction number matrices.
    """

    collection = dataCollection()
    
    # loop for each day
    
    for i in range(sus_portion.shape[1]):

        eff_rep_num_mat = distRepNum.from_epi_anl(basic_rep_num_mat, sus_portion[:, i], inf_portion[:, i], population)
        collection.add(eff_rep_num_mat)
    
    return collection

def compute_eff_rep_num_sec_clu(M : int,
                                population: np.ndarray,
                                firstClu2SecondClu: dict, 
                                inf_portion: np.ndarray, 
                                eff_rep_num_mat_list: list,
                                clip_value: int = 20) -> dataCollection:
    """
    Get the distributed effective reproduction number after the second clustering

    Input:
        - M: int, number of clusters after the second clustering
        - population: numpy.ndarray. Population of clusters. N by 1, where N -> number of clusters.
        - firstClu2SecondClu: dictionary, the map from the cluster_id after the first clustering 
            to the cluster_id after the second clustering. Length: N -> the number of clusters after first clustering
        - sus_portion: suspected portion after the first clustering. Dimension: N by D.
            M -> the number of clusters after first clustering, D -> number of dates.
        - eff_rep_num_mat_list: a list of effective repreductive number matrices with length D
        - clip_value: int. A cap for reproduction number.

    Output:
        - res: a list of effective repreductive number matrices after second clustering with length D.
    """
    N = len(firstClu2SecondClu) # the number of clusters after first clustering
    D = inf_portion.shape[1]  # number of days
    
    # initialize
    res = dataCollection()
    
    for d in range(D):
        eff_mat = np.zeros((M,M))
        eff_mat_prev = eff_rep_num_mat_list[d].eff_rep_num_mat
        denomiator = np.zeros(M)
        
        for x in range(N):
            for y in range(N):
                map_x = firstClu2SecondClu[x]
                map_y = firstClu2SecondClu[y]
                
                if (inf_portion[x][d] == 0):
                    # if the infected portion is 0, we assume at lease one person is
                    # infected to prevent computational problem
                    eff_mat[map_x][map_y] += (1.0 / population[x]) * eff_mat_prev[x][y]
                else:
                    eff_mat[map_x][map_y] += inf_portion[x][d] * eff_mat_prev[x][y]

            if (inf_portion[x][d] == 0):
                # Similarly, we assume at lease one person is
                # infected to prevent computational problem
                denomiator[map_x] += (1.0 / population[x])
            else:
                denomiator[map_x] += inf_portion[x][d]

        for i in range(M):
            eff_mat[i, :] = eff_mat[i, :] / denomiator[i]

        eff_mat = np.clip(eff_mat, 0, clip_value)
        res.add(distRepNum(eff_mat))

    return res

def id_to_plot(df_clusters: pd.DataFrame, id: int):
    """
    Plot all clusters, and given a cluster id, mark it on the map.

    Input:
        - df_clusters: dataframe of the basic infomation of clusters
        - id: cluster_id
    """

    # locations
    X = df_clusters['X'].to_numpy()
    Y = df_clusters['Y'].to_numpy()

    npts = len(X)
    indices = np.arange(npts)

    norm = mco.Normalize(vmin=np.min(indices), vmax=np.max(indices))
    cmap = cm.rainbow # type:ignore

    colors = cmap(norm(indices))

    fig = plt.figure(figsize=(4,3),dpi=300)
    ax = fig.gca()
    for i in np.arange(npts):
    #     if i % idump == 0:
    #         print(f"{i} / {npts}")
        x = X[i]
        y = Y[i]
        circle = plt.Circle((x,y), 0.5, color=colors[i], alpha=0.5, lw=0)   # type:ignore
        ax.add_patch(circle)

    ###########################################################
    ##### Add an rectangle on the cluster that we want to mark.
    ###########################################################
    print(f"This location has longitude {X[id]} and latitude {Y[id]}.")
    rect = Rectangle(xy=[X[id]-1, Y[id]-1], width=2, height=2, edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
        
    xmin = np.min(X) - 0.5
    xmax = np.max(X) + 0.5
    ymin = np.min(Y) - 0.5
    ymax = np.max(Y) + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    for lab in 'left', 'right', 'bottom', 'top':
        ax.spines[lab].set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cax = fig.add_axes(rect=[0.98,0.1,0.02,0.7])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, label='Matrix index', extendfrac='auto')
    plt.show()
    return

def plot_dist_rep(eff_rep_num_list: list,
                  target_id: int,
                  clusters_id_list: list):
    """
    
    """
    # days 
    D = len(eff_rep_num_list)
    length = len(clusters_id_list)

    res = []
    for _ in range(length):
        res.append([])

    for i in range(D):
        for j in range(length):
            res[j].append(eff_rep_num_list[i].eff_rep_num_mat[target_id][clusters_id_list[j]])

    for i in range(length):
        plt.plot(res[i][100:300])

    plt.legend()
    plt.show()

def get_rep_num_time_series(eff_rep_num_list: list, in_id: int, out_id: int) -> list:
    """
    Get effective reproduction number given in-node id and out-node id.
    """
    # days 
    D = len(eff_rep_num_list)
    
    res = []

    for i in range(D):
        res.append(eff_rep_num_list[i][in_id,out_id])

    return res

def get_portion_time_series(df:pd.DataFrame, id:int):

    """
    Get portion data, which is a time series data, given node id.
    """
    return df.loc[id].to_numpy()


