from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

K = 16          #No. of clusters/colours

def main():
    path_large = '../data/peppers-large.tiff'
    path_small = '../data/peppers-small.tiff'
    A_small = imread(path_small)
    A_large = imread(path_large)
    A_large = np.array(A_large)
    mu = k_means(A_small)
    m,n,c = A_large.shape

    weights = []
    for i in range(K):
        weights.append(np.linalg.norm(A_large - mu[i],axis = -1))
    
    cluster = np.argmin(weights,axis = 0) 

    for i in range(m):
        for j in range(n):
            A_large[i][j] = mu[cluster[i][j]]
    plt.imshow(A_large)
    plt.show()

def k_means(A):
    m,n,c = A.shape
    rand_1 = np.random.choice(np.arange(m),K)
    rand_2 = np.random.choice(np.arange(n),K)
    mu = A[rand_1,rand_2]
    it = 0
    error = 1
    while it < 500 and error > 1e-4:
        it += 1
        weights = []
        mu_sum = np.zeros([K,c])
        cluster_count = np.zeros(K)
        for i in range(K):
            weights.append(np.linalg.norm(A - mu[i],axis = -1))
        cluster = np.argmin(weights,axis = 0)

        for i in range(m):
            for j in range(n):
                mu_sum[cluster[i][j]] += A[i][j]
                cluster_count[cluster[i][j]] += 1
        mu_new = mu_sum/np.reshape(cluster_count,(K,1))
        error = np.linalg.norm(mu-mu_new)
        mu = mu_new
        print("Iteration - ",it," - ",error)

    return mu

if __name__ == '__main__':
    main()