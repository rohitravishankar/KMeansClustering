import numpy as np
from scipy import misc

def kMeans(points, k, maxIter = 10):
    """returns k means clustered points"""
    
    # inits centroids randomly
    centroids = initialize_centroids(points, k)

    #iterate as many times as we should
    for i in range(0, int(maxIter)):
        print("Iteration : " + str(i  + 1))
        closestCentroids = closest_centroids(points, centroids)
        centroids = move_centroids(points, closestCentroids, centroids)

        #return new points
        finalPoints = set_to_centroids( points, centroids, closestCentroids)
    return finalPoints

def initialize_centroids(points, k ):
    """:returns k centroids from the initial points"""
    (x, y, z) = points.shape
    centroids = points.copy().reshape(x*y, z)
    np.random.shuffle(centroids)
    return centroids[:k]

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    newCentroids = []

    #iterate over all k
    for i in range(centroids.shape[0]):

        # find which points are in group 1
        # returns a matrix of true/false for all points
        indices = (closest==i)

        # gets all the points
        correspondingPoints = points[indices]

        # get the average
        average = correspondingPoints.mean(axis=0)

        # add our new centroids
        newCentroids.append(average)

    return np.array(newCentroids)

def set_to_centroids(points, centroids, closestCentroids):
    """returnds matrix of all points set to the value of their corresponding centroids"""

    # Make a matrix to hold new points
    newPoints = np.zeros(points.shape)

    #set all points to corresponding centroids
    (x, y) = closestCentroids.shape

    for i in range(x):
        for j in range(y):
            newPoints[i][j] = centroids[closestCentroids[i][j]]

    # newPoints = centroids[closestCentroids]
    return newPoints


def closest_centroids( points, centroids ):
    """returns an array containing the index to the nearest centroids for each points"""

    # Get differences between each point and all centroids (element wise subtraction)
    # Each row here will be an array of differences between pixels
    differences = points - centroids[:, np.newaxis, np.newaxis]

    # Square the differences
    squareDifference = differences ** 2

    # Sum rgb differences
    summedSquaredDifferences = squareDifference.sum(axis = 3)

    # Square root distances
    finalDistances = np.sqrt(summedSquaredDifferences)

    # return vector of which centroids are closest
    return np.argmin(finalDistances, axis=0)
    

def main():
    
    # get input
    imageName = input('Please enter image name: ')
    points = misc.imread(imageName)
    k = int(input("Please enter k:"))
    
    # get new mateix, save it
    newPoints = kMeans(points, k)
    
    # Save new pixels to new image
    misc.imsave('outFile.jpg', newPoints)

if __name__ == '__main__':
    main()