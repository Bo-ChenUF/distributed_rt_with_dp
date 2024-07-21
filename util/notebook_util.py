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

def binary_search_beta(start, end, target, Rec_matrix_inv, contact_matrix):
    """
        Search for beta such that the resulting reproduction number matches the 
        given reproduction number.

        Input:
            - start: starting point
            - end: ending point
            - target: target reproduction number
            - Rec_matrix_inv: inverse recovery matrix
            - contact matrix: contact matrix
        
        Output:
            - beta: value of beta
    """

    # corner case:
    if spectral_radius(end * Rec_matrix_inv @ contact_matrix) < target or spectral_radius(start * Rec_matrix_inv @ contact_matrix) > target:
        raise ValueError("The target value is not in the given range! Please raise the value of 'end'")

    tolerance = 0
    recursive_max = 100
    count = 0

    while start < end or count <= recursive_max:
        # middle point
        mid = start + (end-start) / 2

        product_matrix =  mid * Rec_matrix_inv @ contact_matrix
        radius = spectral_radius(product_matrix)

        # check if the target value is reached.
        if (abs(target - radius) <= tolerance): 
            break

        if radius < target:
            start = mid # Increase beta
        else:
            end = mid  # Decrease beta

        count += 1

    # The following lines will be triggered if the max number 
    # of recursions reached.
    if (count > recursive_max):
        print("Max number of recursions reached!")

    return mid




        
