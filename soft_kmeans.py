# ID: 20180373 NAME: Kim Hyeonji
######################################################################################

import os
import math
from utils import converged, plot_2d_soft, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt
from kmeans import euclidean_distance

# problem for students
def get_responsibility(data_point, centroids, beta):
    """Calculate the responsibiliy of each cluster for a single data point.
    You should use the euclidean_distance function (that you previously implemented).
    You can use the math.exp() function to calculate the responsibility.

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations
        beta: hyper-parameter

    Returns: a dictionary whose keys are the the centroids' key names and
             value is a float as the responsibility of the cluster for the data point.
    """
    responsibility_dict = dict()
    sum_responsibility = 0
    for centroid in centroids.items() :
        sum_responsibility += math.exp(-beta * euclidean_distance(data_point, centroid[1]))
        
    for centroid in centroids.items() :
        responsibility = math.exp(-beta * euclidean_distance(data_point, centroid[1]))/sum_responsibility 
        responsibility_dict[centroid[0]] = responsibility
    return responsibility_dict


# problem for students
def update_soft_assignment(data, centroids, beta):
    """Find the responsibility of each cluster for all data points.
    You should use the get_responsibility function (that you previously implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a dictionary whose keys are the data points of type 'tuple'
             and values are the dictionary returned by get_responsibility function.
             (In python, 'list' cannot be the 'key' of 'dict')
             
    """
    
    soft_assignment_dict = dict()
    for data_point in data:
        responsibility_dict = get_responsibility(data_point, centroids, beta)
        soft_assignment_dict[tuple(data_point)] = responsibility_dict
    return soft_assignment_dict
            

# problem for students
def update_centroids(soft_assignment_dict):
    """Update centroid locations with the responsibility of the cluster for each point
    as a weight. You can numpy methods for simple array computations. But the values of 
    the result dictionary must be of type 'list'.

    Arguments:
        assignment_dict: the dictionary returned by update_soft_assignment function

    Returns: A new dictionary representing the updated centroids
    """
    centroids = dict()
    sum_responsibility = dict()
    sum_responsibility_datapoint = dict()
    
    for soft_assignment in soft_assignment_dict.items():
        data_point = soft_assignment[0]
        for centroid_name, responsibility in soft_assignment[1].items():
            if centroid_name not in sum_responsibility.keys():
                sum_responsibility[centroid_name] = responsibility
            else :
                sum_responsibility[centroid_name] += responsibility
            if centroid_name not in sum_responsibility_datapoint.keys():
                sum_responsibility_datapoint[centroid_name] = tuple([x*responsibility for x in data_point])
            else : 
                sum_responsibility_datapoint[centroid_name] = tuple(map(sum, zip(sum_responsibility_datapoint[centroid_name], tuple([x*responsibility for x in data_point])))) 
    
    for centroid_name in list(sum_responsibility.keys()):
        centroids[centroid_name] =[x/sum_responsibility[centroid_name] for x in sum_responsibility_datapoint[centroid_name]]
        
    return centroids

def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    beta = 3
    centroids = init_centroids
    old_centroids = None
    total_step = 7
    for step in range(total_step):
        # save old centroid
        old_centroids = centroids
        # new assignment
        soft_assignment_dict = update_soft_assignment(data, old_centroids, beta)
        # update centroids
        centroids = update_centroids(soft_assignment_dict)
        # plot centroid
        fig = plot_2d_soft(soft_assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D_soft", f"step{step}.png"))
        plt.clf()
    print(f"{total_step} iterations were completed.")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/data_2d.csv")
    init_c = load_centroids("data/2d_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("2d_final_centroids_with_soft_kmeans.csv", final_c)
