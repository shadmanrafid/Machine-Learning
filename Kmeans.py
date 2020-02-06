from Blackbox41 import Blackbox41
from Blackbox42 import Blackbox42
import sys
import numpy as np

def kmeans(data, k=4, normalize=False, limit=4000):
    # Normalizing the data
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]
    # Taking random centroids
    centers = data[:k]

    for i in range(limit):
        # Assigning the data values to different clusters
        classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
        # Recalculating the Centroids
        new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

        if (new_centers == centers).all():
            break
        else:
            centers = new_centers

    if normalize:
        centers = centers * stats[1] + stats[0]

    return classifications, centers

if __name__ == '__main__':

    Input_file = sys.argv[-1]
    # Taking Input
    Input = list()
    if Input_file == 'blackbox41':
        bb = Blackbox41()
        Input = bb.ask()
    elif Input_file == 'blackbox42':
        bb = Blackbox42()
        Input = bb.ask()
    else:
        print('invalid blackbox')
        sys.exit()
    # Creating the clusters
    classifications, centers = kmeans(Input, normalize=True, k=4)
    # Creating the output file
    output_file = 'results_'
    if Input_file == 'blackbox41':
        output_file += 'blackbox41.csv'
    elif Input_file == 'blackbox42':
        output_file += 'blackbox42.csv'

    classifications.tofile(output_file, sep='\n')