import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score, v_measure_score
from sklearn.decomposition import TruncatedSVD, PCA

import time

np.random.seed(0)


def calculate_distances(x1, x2):
    """
    compute the euclidean distance
    :param x1: array like
    :param x2: array like
    :return: euclidean distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, X, n_clusters=3, iteration_limit=100, init='random'):
        self.start = time.process_time()
        self.init = init
        self.K = n_clusters
        self.max_iterations = iteration_limit
        self.X_train = X
        self.values, self.features = X.shape
        self.centroids = None
        self.labels = np.empty(self.values)
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        if init == 'k++':
            self.centroids = self.__kmeans_init__()
        else:
            self.centroids = self.__random_init__()

    def __random_init__(self):
        """
        Choose a random K centroids from the sample
        :return: K centroids
        """
        choice = np.random.choice(self.X_train.shape[0], self.K, replace=False)
        return self.X_train[choice, :]

    def __kmeans_init__(self):
        """
        initializes centroids using k means ++.
        :return: K centroids
        """
        loc = np.random.choice(self.values)  # initialize with a random choice for the first centroid.
        centroid = [self.X_train[loc]]
        # loop for the remaining K-1 times and calculate the distance to pick the right location for centroid
        for _ in range(self.K - 1):
            distances = [calculate_distances(self.X_train, target) for target in centroid]
            centroid.append(self.X_train[np.argmax(np.argmin(distances, axis=0))])
        return np.array(centroid)

    def __build_clusters__(self):
        """
        Generate clusters by using the closest centroids to the cluster (samples)
        :return: the new cluster generated.
        """
        clusters = [[] for _ in range(self.K)]
        for i, sample in enumerate(self.X_train):
            # find the closest location to the cluster
            distances = [calculate_distances(sample, loc) for loc in self.centroids]
            loc = np.argmin(distances)
            clusters[loc].append(i)
        return clusters

    def __check_convergence__(self, centroids_old, centroids, tolerance=0.05):
        """
        check whether the old centroid and new are different or not.
        :param centroids_old: previous centroid location.
        :param centroids: new centroid location.
        :param tolerance: limit of tolerance of closeness.
        :return: true if converged or false if not.
        """
        # distances between each old and new centroids, fol all centroids
        distances = [calculate_distances(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) <= tolerance

    def predict(self):
        """
        Find the best clusters and centroids based on K.
        :return: location indices of the clusters.
        """
        for j in range(self.max_iterations):
            # Assign samples to the closest centroids (create clusters)
            self.clusters = self.__build_clusters__()
            centroids_old = self.centroids  # save previous centroid
            # generate new centroids by calculating the mean of each of the clusters.
            for loc, cluster in enumerate(self.clusters):
                self.centroids[loc] = np.mean(self.X_train[cluster], axis=0)
            if self.__check_convergence__(centroids_old, self.centroids):  # convergence check
                break

        for indices, clusters in enumerate(self.clusters):
            for loc in clusters:
                self.labels[loc] = indices
        print("K-means algorithm using {} initialization took {} seconds to run.".
              format(self.init, time.process_time() - self.start))
        return self.labels


dataset = load_iris()
X = dataset.data
Y = dataset.target

# use standard scaler for preprocessing the iris data.
# sc = StandardScaler()
# X = sc.fit_transform(X)

# Applying k-means to the dataset
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
score = []
score2 = []
v_score = []
v_score2 = []
print("applying k means for different k values...\n")
for k in K:
    kmeans = KMeans(X, n_clusters=k, iteration_limit=100)
    y_pred = kmeans.predict()
    score.append(silhouette_score(X, y_pred))
    v_score.append(v_measure_score(Y, y_pred))

# --- uncomment the following lines to plot the results ----0
# plt.plot(K, v_score, label='v_score', color='red')
# plt.plot(K, score, label='silhouette', color='blue')
# plt.xlabel('K')
# plt.ylabel("score")
# plt.title("IRIS dataset")
# plt.legend(loc='best')
# plt.savefig("kmeans_iris_score.png")
# ------------------------------------------------------------

print("K means on IRIS dataset is completed...")
print("Silhouette scores for k values from 2 to 10\n", score)
print("V scores for k values from 2 to 10\n", v_score)
print("K means on IRIS completed.\n")

# -------------------------------------------------------------------------------------------
# Uncomment the following line to see a visual illustration of clusters on the IRIS dataset.
# visualizing the clusters
# kmeans = KMeans(X, n_clusters=7, iteration_limit=100)
# y_pred = kmeans.predict()
# for i, index in enumerate(kmeans.clusters):
#     plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=100, cmap='rainbow', label='Cluster {}'.format(i))
# plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=100, c='black', label='Centroids')
#
# plt.title('Clusters of Iris Dataset')
# plt.xlabel('features')
# plt.ylabel('labels')
# plt.legend(loc='best')
# plt.savefig("cluster_7.png")


Predictions = "mnist_clusters.txt"
figure = "mnist_silhouette.png"
print("\nOpening MNIST test file...")
with open("test_data.txt") as f:
    X_test = np.loadtxt(f, delimiter=',')
print("#" * 20, "MNIST Dataset", "#" * 20)

pca = PCA(n_components=100)
x = pca.fit_transform(X_test)
# svd = TruncatedSVD(n_components=10)
# x = svd.fit_transform(X_test)
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
score = []  # silhouette score
# y_pred2 = None
# for k in K:
#     kmeans2 = KMeans(x, n_clusters=k, iteration_limit=100)
#     y_pred2 = kmeans2.predict()
#     score.append(silhouette_score(X_test, y_pred2))
# # save predictions to file for K = 10
# with open(Predictions, "w") as f:
#     for label in y_pred2:
#         f.write(str(int(label)) + "\n")
# print("Cluster prediction Done")
# print("Silhouette score: {:.4f} for K = 10".format(score[8]))

# plt.plot(K, score, label='silhouette', color='blue')
# plt.xlabel('K')
# plt.ylabel("score")
# plt.title("MNIST dataset")
# plt.legend(loc='best')
# plt.savefig(figure)
#
# colors = ['red', 'green', 'blue', 'yellow', 'pink', 'cyan', 'brown', 'purple', 'grey']
kmeans2 = KMeans(x, n_clusters=10, iteration_limit=100)
y_pred2 = kmeans2.predict()
print(len(kmeans2.clusters[0]))
for i, index in enumerate(kmeans2.clusters):
    plt.scatter(x[y_pred2 == i, 0], x[y_pred2 == i, 1], s=100, cmap='rainbow', label='Cluster {}'.format(i))
plt.scatter(kmeans2.centroids[:, 0], kmeans2.centroids[:, 1], s=100, c='black', label='Centroids')

plt.title('Clusters of Iris Dataset')
plt.xlabel('cluster size')
plt.ylabel('features')
plt.legend(loc='best')
plt.show()
# plt.savefig("mnist_clusters.png")
