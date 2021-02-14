# This is a pure copy paste of the EM_Image_Segmentation notebook

import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


# Define max number of iterations
EPOCHS = 30
# Define convergence tolerance
tol = 1e-4

path_to_image = '../images/img.jpg'

# Let's say we want to find 8 clusters in the image
K = 32


def image_to_data(image_data):
    '''
    Convert 3D image to 2D. From WxHxD to (WxH)xD.
    :param image_data: The image you want to convert
    :return: The new 2D image
    '''
    height, width, D = image_data.shape
    N = height * width
    X = np.zeros((N, D))
    for w in range(width):
        for h in range(height):
            n = h + (w - 1) * height
            X[n, 0] = image_data[h, w, 0]
            X[n, 1] = image_data[h, w, 1]
            X[n, 2] = image_data[h, w, 2]

    return X


def data_to_image(X, height, width):
    '''
    Convert the NxD matrix to WxHxD where WxH = N
    :param X: The image you want to convert to 3D
    :param height: The height your image should have
    :param width: The wifth your image should have
    :return: The new image in 3 dimensions
    '''
    N, D = X.shape
    newImage = np.zeros((height, width, D))
    for n in range(1, N+1):
        w = np.fix(n/height)
        if n % height != 0:
            w = w + 1
        h = n - (w - 1) * height
        newImage[int(h)-1, int(w)-1, :] = X[n-1, :]

    return newImage


# Get the image as a N X D matrix.
start_img = Image.open(path_to_image, 'r')
img_array = np.array(start_img)
X = image_to_data(img_array)
# Normalize pixel values
X = X/np.max(X)
# X = X/255
# get image properties.
height, width, D = img_array.shape

# Get matrix properties.
N, D = X.shape

# Initialize the probabilities of gamma(Ζnk) as a numpy array of shape NxK filled with zeroes.
gamma = np.zeros((N, K))

# Initialize p (1 X K vector). It keeps the prior probality of cluster k.
p = np.zeros(K)
p[:K] = 1/K

# Initialize the m, K X D vector.
# m_k is the centroid of cluster k.
# Initialize m, using K random points from X.
m = X[np.random.randint(X.shape[0], size=K), :]

# The initialization value of sigma is essential on how the algorithm performs so tune it to your needs.
sigma = (0.1 * np.var(X)) * np.ones(K)


def maximum_likelihood(X, m, K, sigma, p):
    """ This function is used in order to compute the maximum likelihood using numerical stable way.
    :param X: The 2D representation of the image.
    :param m: A vector of length K with the cluster centers.
    :param K: Number of clusters we want to find.
    :param sigma: The covariance matrix which is a vector of length K.
    :param p: The apriori distribution.
    :return : The maximum log likelihood for the given parameters.
    """
    N, D = X.shape
    lhood = np.zeros((N, K))
    for k in range(K):
        for d in range(D):
            lhood[:, k] = lhood[:, k] + np.log(np.sqrt(2*np.pi*sigma[k])) + (((X[:, d] - m[k, d])*(X[:, d] - m[k, d])) / (2*sigma[k]))

        lhood[:, k] = np.log(p[k]) - lhood[:, k]

    maxF = np.amax(lhood, axis=1)
    repMax = np.tile(maxF, (K, 1)).T
    lhood = np.exp(lhood - repMax)
    # Convert to log likelihood
    likelihood = np.sum(maxF + np.log(np.sum(lhood, axis=1)))

    return likelihood

# Loop until convergence, or if maximum iterations are reached.
for e in tqdm(range(EPOCHS)):
    ##################################### EXPECTATION STEP #####################################
    s = np.zeros((N, K))
    for k in range(K):
        for d in range(D):
            s[:, k] = s[:, k] + np.log(np.sqrt(2*np.pi)*sigma[k]) + (((X[:, d] - m[k, d])*(X[:, d] - m[k, d])) / (2*sigma[k]))
        s[:, k] = np.log(p[k]) - s[:, k]

    s = np.exp(s)
    # Update gamma
    gamma = np.divide(s, (np.tile(np.sum(s, axis=1), (K, 1))).T)

    ##################################### MAXIMIZATION STEP #####################################
    sum_of_gamma = np.sum(gamma, axis=0)
    sk = np.zeros(K)
    for k in range(K):

        for d in range(D):
            m[k, d] = np.divide(np.dot(gamma[:, k].T, X[:, d]), sum_of_gamma[k])  # Calculate mean_new of k, d

        # Update sigma for each k using the new m
        inside_of_sigma = np.sum(np.multiply(X-m[k, :], X-m[k, :]), axis=1)
        sk[k] = np.sum(gamma[:, k]*inside_of_sigma)
        denom = np.sum(gamma[:, k])
        sk[k] = sk[k]/(D*denom)

    p = sum_of_gamma / N
    sigma = sk
    # Keep only significant values
    # sigma[sigma < 1e-06] = 1e-06

    ##################################### CHECK FOR CONVERGENCE #####################################
    # likelihoodNew = maximum_likelihood(X, m, K, sigma, p)

# Assign each data point to its closer cluster
maxK = np.argmax(gamma, axis=1)  # Nx1 for each pixel
# Get the new image where each pixel has the value of its closest center
clusteredX = m[maxK, :]

# Start plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
# Convert image to 3D
threed_img = data_to_image(clusteredX, height, width)
ax1.imshow(threed_img)
ax1.set_title('[Image with K={} colors] Segmentation by EM'.format(K))
# Set x-axis, y-axis
ax1.axes.set_xlabel('x-coordinate')
ax1.axes.set_ylabel('y-coordinate')

ax2.imshow(start_img)
ax2.set_title("[Initial Image]")
# Set x-axis, y-axis
ax2.axes.set_xlabel('x-coordinate')
ax2.axes.set_ylabel('y-coordinate')

plt.savefig("test_output.jpg")
plt.show()
