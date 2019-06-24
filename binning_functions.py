# Imported from the Jacobs RSage
# https://github.com/jacobseiler/rsage/blob/master/output/CollectiveStats.py

import numpy as np
import scipy


def calculate_pooled_stats(rank, comm, mean_pool, std_pool, N_pool, mean_local, std_local, N_local):
    '''
    Calculates the pooled mean and standard deviation from multiple processors and appends it to an input array.
    Formulae taken from https://en.wikipedia.org/wiki/Pooled_variance
    As we only care about these stats on the rank 0 process, we make use of junk inputs/outputs for other ranks.
    NOTE: Since the input data may be an array (e.g. pooling the mean/std for a stellar mass function).
    Parameters
    ----------
    rank: Integer
        Rank of the task.
    mean_pool, std_pool, N_pool : array of floats.
        Arrays that contain the current pooled means/standard deviation/number of data points (for rank 0) or just a junk input (for other ranks).
    mean_local, mean_std : float or array of floats.
        The non-pooled mean and standard deviation unique for each process.
    N_local : floating point number or array of floating point numbers. 
        Number of data points used to calculate the mean/standard deviation that is going to be added to the pool.
        NOTE: Use floating point here so we can use MPI.DOUBLE for all MPI functions.
    Returns
    -------
    mean_pool, std_pool : array of floats.
        Original array with the new pooled mean/standard deviation appended (for rank 0) or the new pooled mean/standard deviation only (for other ranks).
    Units
    -----
    All units are the same as the input.
    All inputs MUST BE real-space (not log-space).
    '''

    if isinstance(mean_local, list) == True:    
        if len(mean_local) != len(std_local):
            print("len(mean_local) = {0} \t len(std_local) = {1}".format(len(mean_local), len(std_local)))
            raise ValueError("Lengths of mean_local and std_local should be equal")
   
    if ((type(mean_local).__module__ == np.__name__) == True or (isinstance(mean_local, list) == True)): # Checks to see if we are dealing with arrays. 
    
        N_times_mean_local = np.multiply(N_local, mean_local)
        N_times_var_local = np.multiply(N_local, np.multiply(std_local, std_local))
        
        N_local = np.array(N_local).astype(float)
        N_times_mean_local = np.array(N_times_mean_local).astype(np.float32)

        if rank == 0: # Only rank 0 holds the final arrays so only it requires proper definitions.
            N_times_mean_pool = np.zeros_like(N_times_mean_local) 
            N_pool_function = np.zeros_like(N_local)
            N_times_var_pool = np.zeros_like(N_times_var_local)

            N_times_mean_pool = N_times_mean_pool.astype(np.float64) # Recast everything to double precision then use MPI.DOUBLE.
            N_pool_function = N_pool_function.astype(np.float64)
            N_times_var_pool = N_times_var_pool.astype(np.float64)
        else:
            N_times_mean_pool = None
            N_pool_function = None
            N_times_var_pool = None


        N_times_mean_local = N_times_mean_local.astype(np.float64)
        N_local = N_local.astype(np.float64)
        N_times_var_local = N_times_var_local.astype(np.float64)

        comm.Reduce([N_times_mean_local, MPI.DOUBLE], [N_times_mean_pool, MPI.DOUBLE], op = MPI.SUM, root = 0) # Sum the arrays across processors.
        comm.Reduce([N_local, MPI.DOUBLE],[N_pool_function, MPI.DOUBLE], op = MPI.SUM, root = 0)   
        comm.Reduce([N_times_var_local, MPI.DOUBLE], [N_times_var_pool, MPI.DOUBLE], op = MPI.SUM, root = 0)
        
    else:
    
        N_times_mean_local = N_local * mean_local
        N_times_var_local = N_local * std_local * std_local

        N_times_mean_pool = comm.reduce(N_times_mean_local, op = MPI.SUM, root = 0)
        N_pool_function = comm.reduce(N_local, op = MPI.SUM, root = 0)
        N_times_var_pool = comm.reduce(N_times_var_local, op = MPI.SUM, root = 0)
    
    if rank == 0:

        mean_pool_function = np.zeros((len(N_pool_function)))
        std_pool_function = np.zeros((len(N_pool_function)))

        for i in range(0, len(N_pool_function)):
            if N_pool_function[i] == 0:
                mean_pool_function[i] = 0.0
            else:
                mean_pool_function[i] = np.divide(N_times_mean_pool[i], N_pool_function[i])
            if N_pool_function[i] < 3:
                std_pool_function[i] = 0.0
            else:
                std_pool_function[i] = np.sqrt(np.divide(N_times_var_pool[i], N_pool_function[i]))
       
        mean_pool.append(mean_pool_function)
        std_pool.append(std_pool_function)
        N_pool.append(N_pool_function)

        return mean_pool, std_pool, N_pool
    else:
    
        return mean_pool, std_pool, N_pool_function # Junk return because non-rank 0 doesn't care.


# https://github.com/jacobseiler/rsage/blob/master/output/GalaxyData.py
def do_2D_binning(data_x, data_y, data_to_bin, bins_y, bins_x):
    """
    Calculate the mean and number of points in a grid defined by [bins_x, bins_y].

    Parameters
    ----------
    data_x, data_y : Numpy-arrays of floats 
        Data that we are using to update the histograms.

    data_to_bin : ...

    bins : Numpy-array of floats
        The bins we are binning the y-data on.  Defined in units/properties of
        the x-data.

    Returns
    ---------

    mean_binned, N_binned : Numpy-arrays of floats with size [N_x_bins, N_y_bins]  **This may change depennding on your definition 'bin'**
        The mean and number of data points in each bin.
    """

    # The bins should be a list...
    bins = np.array((bins_x, bins_y))

    print(bins)

    for array in [data_x, data_y, data_to_bin, bins]:
        print(array.shape)
        print(type(array))

    #tmp_x = data_x[:,0]
    #tmp_y = data_y[:,0]
    #tmp_data = data_to_bin[:,0]

    mean_binned, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(data_x, data_y, data_to_bin,
                                                  statistic='mean', bins=bins)

    # Do N_binned...

    return mean_binned, xedges, yedges, binnumber#, N_binned
