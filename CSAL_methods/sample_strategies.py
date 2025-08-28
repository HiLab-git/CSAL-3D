import numpy as np
import random
import os.path as osp
from scipy.special import softmax
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

def select_valuable_samples_FPS(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering and Euclidean distance.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    num_clusters = num_samples // 2  # Number of clusters for KMeans

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(feats)

    labels = kmeans.labels_

    select_samples_list = []

    extra_sample_needed = num_samples % 2 == 1

    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_feats = feats[cluster_indices]

        distances = cdist(cluster_feats, cluster_feats, metric='euclidean')

        max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)

        sample1_index = cluster_indices[max_dist_indices[0]]
        sample2_index = cluster_indices[max_dist_indices[1]]

        select_samples_list.append(name_list[sample1_index])
        select_samples_list.append(name_list[sample2_index])

    if extra_sample_needed:
        last_cluster_indices = np.where(labels == num_clusters - 1)[0]
        select_samples_list.append(name_list[last_cluster_indices[0]])

    return select_samples_list[:num_samples]  

def generate_FPS_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_FPS(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/FPS_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def compute_typicality(cluster_feats):
    """
    Compute typicality for each sample in the cluster.

    Parameters:
    - cluster_feats: ndarray of shape (n_c, p), feature vectors in the cluster.

    Returns:
    - typicalities: ndarray of shape (n_c,), typicality of each sample.
    """

    distances = cdist(cluster_feats, cluster_feats, metric='euclidean')
    avg_distances = np.mean(distances, axis=1)
    typicalities = 1 / avg_distances

    return typicalities

def select_valuable_samples_TypiClust(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering and typicality.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    kmeans.fit(feats)

    labels = kmeans.labels_
    select_samples_list = []


    for i in range(num_samples):
        cluster_indices = np.where(labels == i)[0]
        cluster_feats = feats[cluster_indices]

        typicalities = compute_typicality(cluster_feats)
        highest_typicality_index = cluster_indices[np.argmax(typicalities)]

        select_samples_list.append(name_list[highest_typicality_index])

    return select_samples_list

def generate_TypiClust_plan(organ, feats_file, num_samples):
    np.random.seed(1001)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_TypiClust(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/TypiClust.npz'
    np.savez(save_path, paths=paths)

def compute_information_density(cluster_feats):
    """
    Compute information density for each sample in the cluster.

    Parameters:
    - cluster_feats: ndarray of shape (n_c, p), feature vectors in the cluster.

    Returns:
    - densities: ndarray of shape (n_c,), information density of each sample.
    """

    similarity_matrix = cosine_similarity(cluster_feats)
    densities = similarity_matrix.mean(axis=1)

    return densities

def select_valuable_samples_CALR(feats, name_list, num_samples):
    """
    Select valuable samples using BIRCH clustering and information density.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    # Perform BIRCH clustering
    birch = Birch(n_clusters=num_samples)
    birch.fit(feats)

    labels = birch.labels_

    select_samples_list = []

    for i in range(num_samples):
        cluster_indices = np.where(labels == i)[0]
        cluster_feats = feats[cluster_indices]
        densities = compute_information_density(cluster_feats)

        highest_density_index = cluster_indices[np.argmax(densities)]

        select_samples_list.append(name_list[highest_density_index])

    return select_samples_list

def generate_CALR_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_CALR(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/CALR_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def select_valuable_samples_ALPS(feats, name_list, num_samples):
    """
    Select valuable samples using KMeans clustering.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    kmeans.fit(feats)

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Get the labels of each point
    labels = kmeans.labels_

    select_samples_list = []

    for i in range(num_samples):
        cluster_indices = np.where(labels == i)[0]

        distances = np.linalg.norm(feats[cluster_indices] - centers[i], axis=1)

        closest_index = cluster_indices[np.argmin(distances)]

        select_samples_list.append(name_list[closest_index])

    return select_samples_list

def generate_ALPS_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_ALPS(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/ALPS.npz'
    np.savez(save_path, paths=paths)

def read_tsv(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            scores.append(float(line.strip()))
    return np.array(scores)

def construct_kernels(X, uncertainties):
    gamma_values = [0.1, 1, 10]
    kernels = []
    for gamma in gamma_values:
        pairwise_dist = pairwise_distances(X, metric='euclidean')
        weights = 1 / (uncertainties[:, np.newaxis] + uncertainties[np.newaxis, :] + 1e-10)
        kernel = np.exp(-gamma * pairwise_dist**2 / 2) * weights
        kernels.append(kernel)
    return kernels

def multiple_kernel_kmeans(kernels, k, max_iter=100):
    n_samples = kernels[0].shape[0]
    cluster_assignments = np.random.choice(k, n_samples)
    
    for _ in range(max_iter):
        combined_kernel = np.zeros((n_samples, n_samples))
        for m in range(len(kernels)):
            combined_kernel += kernels[m]
        
        new_assignments = np.zeros(n_samples)
        for i in range(n_samples):
            distances = np.zeros(k)
            for j in range(k):
                in_cluster = (cluster_assignments == j)
                cluster_size = np.sum(in_cluster)
                if cluster_size == 0:
                    distances[j] = np.inf
                    continue
                cluster_indices = np.where(in_cluster)[0]
                kernel_values = combined_kernel[i, cluster_indices]
                center_dist = combined_kernel[i, i] - 2 * np.sum(kernel_values) / cluster_size + np.sum(combined_kernel[np.ix_(cluster_indices, cluster_indices)]) / (cluster_size ** 2)
                distances[j] = center_dist
            new_assignments[i] = np.argmin(distances)
        
        if np.array_equal(cluster_assignments, new_assignments):
            break
        
        cluster_assignments = new_assignments
    
    centers = np.zeros((k, n_samples))
    for j in range(k):
        in_cluster = (cluster_assignments == j)
        if np.sum(in_cluster) == 0:
            continue
        cluster_indices = np.where(in_cluster)[0]
        centers[j, :] = np.mean(combined_kernel[np.ix_(cluster_indices, cluster_indices)], axis=0)
    
    return centers, cluster_assignments

def calculate_typicality(cluster_points):
    distances = cosine_distances(cluster_points)
    typicality = 1 / (np.mean(distances, axis=1) + 1e-10)
    return typicality

def select_valuable_samples_Uncertainty_Kmeans(feats, score, name_list, num_samples, search_nums):
    uncertainties = np.array(score)
    
    kernels = construct_kernels(feats, uncertainties)
    
    centers, labels = multiple_kernel_kmeans(kernels, num_samples)
    
    selected_indices = []
    for i in range(num_samples):
        cluster_points = feats[labels == i]
        cluster_uncertainties = uncertainties[labels == i]
        cluster_names = name_list[labels == i]

        if len(cluster_points) == 0:
            continue

        typicality = calculate_typicality(cluster_points)
        typical_indices = np.argsort(typicality)[-search_nums:]
        
        neighbors_uncertainties = cluster_uncertainties[typical_indices]
        uncertain_index = typical_indices[np.argmax(neighbors_uncertainties)]
        
        selected_indices.append(np.where(name_list == cluster_names[uncertain_index])[0][0])

    return np.array(selected_indices)

def generate_Uncertainty_Kmeans_plan(organ, feats_file, score_file, num_samples, search_nums):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    score = read_tsv(score_file)
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_Uncertainty_Kmeans(feats=feats, score=score, name_list=name_list, num_samples=num_samples, search_nums=search_nums)

    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in name_list[select_samples_list]]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Ours_{num_samples}.npz'
    np.savez(save_path, paths=paths)
    
def select_valuable_samples_ProbCover(feats, name_list, num_samples, delta=None, alpha=0.95):

    n_samples = feats.shape[0]
    if delta is None:
        num_classes = num_samples 
        delta = estimate_delta(feats, num_classes, alpha)
    
    dist_matrix = distance_matrix(feats, feats)
    adjacency_matrix = (dist_matrix <= delta).astype(int)
    
    selected_samples = []
    
    for _ in range(num_samples):
        out_degrees = adjacency_matrix.sum(axis=1)
        
        max_out_degree_index = np.argmax(out_degrees)
        selected_samples.append(max_out_degree_index)
        
        covered_indices = np.where(adjacency_matrix[max_out_degree_index] > 0)[0]
        adjacency_matrix[:, covered_indices] = 0
        adjacency_matrix[covered_indices, :] = 0 

    return selected_samples

def estimate_delta(embedding, num_classes, alpha=0.95):

    kmeans = KMeans(n_clusters=num_classes)
    cluster_labels = kmeans.fit_predict(embedding)
    
    dist_matrix = distance_matrix(embedding, embedding)
    
    delta_values = np.linspace(0, np.max(dist_matrix), num=100)
    best_delta = 0
    for delta in delta_values:
        pure_balls = 0
        total_balls = 0
        
        for i in range(len(embedding)):
            neighbors = np.where(dist_matrix[i] <= delta)[0]
            if len(neighbors) > 0:
                if np.all(cluster_labels[neighbors] == cluster_labels[i]):
                    pure_balls += 1
                total_balls += 1
        
        purity = pure_balls / total_balls
        if purity >= alpha:
            best_delta = delta
        else:
            break
    
    return best_delta
def generate_Probcover_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])
    
    select_samples_list = select_valuable_samples_ProbCover(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in name_list[select_samples_list]]
    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Probcover.npz'
    np.savez(save_path, paths=paths)

def generate_USL_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_USL(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')

    save_path = f'./{organ}/plans/USL_{num_samples}.npz'
    np.savez(save_path, paths=paths)
    
def generate_USL_T_plan(organ, feats_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_USL_T(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in select_samples_list]

    print(f'Select:\n{paths}')

    save_path = f'./{organ}/plans/USL_T_{num_samples}.npz'
    np.savez(save_path, paths=paths)

def select_valuable_samples_USL(feats, name_list, num_samples, k=5, alpha=2, num_iters=10, lambda_reg=0.5, epsilon=1e-10):
    """
    Select valuable samples using Unsupervised Selective Labeling (USL) with detailed implementation of average distance and inter-cluster regularization.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.
    - k: int, number of nearest neighbors for density calculation.
    - alpha: float, sensitivity hyperparameter for regularization.
    - num_iters: int, number of iterations for regularization.
    - lambda_reg: float, weight for the regularization term.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    n = feats.shape[0]

    distance_matrix = np.linalg.norm(feats[:, np.newaxis, :] - feats[np.newaxis, :, :], axis=2)

    avg_distances = np.zeros(n)
    for i in range(n):
        k_distances = np.sort(distance_matrix[i, :])[1:k + 1]  # Exclude distance to self (first element)
        avg_distances[i] = np.mean(k_distances)

    density = 1 / (avg_distances + epsilon)

    kmeans = KMeans(n_clusters=num_samples, random_state=0)
    kmeans.fit(feats)
    labels = kmeans.labels_

    selected_samples = []
    reg_values = np.zeros(n)

    for iter_count in range(num_iters):
        for i in range(n):
            reg_values[i] = sum(1 / ((np.linalg.norm(feats[i] - feats[j]) ** alpha) + epsilon)
                                for j in selected_samples if len(selected_samples) > 0)

        utility = density - lambda_reg * reg_values
        selected_indices = np.argsort(utility)[-num_samples:]  

        selected_samples = selected_indices.tolist()  

    # Retrieve sample names
    select_samples_list = [name_list[i] for i in selected_samples]
    return select_samples_list

def select_valuable_samples_USL_T(feats, name_list, num_samples, k=5, alpha=2, num_iters=10, lambda_reg=0.5, t=0.25, tau=0.5, epsilon=1e-10):
    """
    Select valuable samples using Training-based Unsupervised Selective Labeling (USL-T) with detailed implementation.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.
    - k: int, number of nearest neighbors for local constraints.
    - alpha: float, sensitivity hyperparameter for regularization.
    - num_iters: int, number of iterations for optimization.
    - lambda_reg: float, weight for the regularization term.
    - t: float, temperature parameter for sharpening.
    - tau: float, threshold for confident assignments in global constraints.
    - epsilon: float, small constant to avoid division by zero.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    n, p = feats.shape
    # Initialize centroids for learnable K-Means
    centroids = feats[random.sample(range(n), num_samples)]
    
    # Initialize soft assignments
    soft_assignments = np.zeros((n, num_samples))
    cluster_confidences = np.zeros(n)
    
    # Neighbor selection for local constraints
    distance_matrix = np.linalg.norm(feats[:, np.newaxis, :] - feats[np.newaxis, :, :], axis=2)
    nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]

    for iteration in range(num_iters):
        similarities = np.dot(feats, centroids.T)
        soft_assignments = softmax(similarities / t, axis=1)
        
        confident_indices = np.max(soft_assignments, axis=1) >= tau
        confident_soft_assignments = soft_assignments[confident_indices]
        confident_feats = feats[confident_indices]
        
        for j in range(num_samples):
            cluster_weights = confident_soft_assignments[:, j]
            weighted_sum = np.sum(cluster_weights[:, np.newaxis] * confident_feats, axis=0)
            centroid_weight = np.sum(cluster_weights)
            centroids[j] = weighted_sum / (centroid_weight + epsilon)
        
        local_regularization = np.zeros((n, num_samples))
        for i in range(n):
            neighbor_indices = nearest_neighbors[i]
            neighbor_assignments = soft_assignments[neighbor_indices]
            neighbor_avg = np.mean(neighbor_assignments, axis=0)
            sharpened_neighbor_avg = softmax(neighbor_avg / t)
            local_regularization[i] = sharpened_neighbor_avg
        
        combined_assignments = lambda_reg * local_regularization + soft_assignments
        cluster_confidences = np.max(combined_assignments, axis=1)
    
    selected_indices = np.argsort(-cluster_confidences)[:num_samples]
    select_samples_list = [name_list[i] for i in selected_indices]
    
    return select_samples_list

def select_valuable_samples_Coreset(feats, name_list, num_samples):
    """
    Select valuable samples using Core-Set (K-Center Greedy) algorithm.

    Parameters:
    - feats: ndarray of shape (n, p), feature vectors.
    - name_list: ndarray of shape (n,), corresponding sample names.
    - num_samples: int, the number of valuable samples to select.

    Returns:
    - select_samples_list: list of selected sample names.
    """
    n_samples = len(name_list)
    labeled_idxs = np.zeros(n_samples, dtype=bool)
    
    dist_mat = np.matmul(feats, feats.T)
    diag = np.diag(dist_mat).reshape(-1, 1)
    dist_mat = -2 * dist_mat + diag + diag.T
    dist_mat = np.sqrt(np.clip(dist_mat, 0, None))

    first_sample_idx = np.random.choice(np.arange(n_samples))
    labeled_idxs[first_sample_idx] = True
    
    min_distances = dist_mat[:, first_sample_idx]

    select_samples_list = [name_list[first_sample_idx]]

    for _ in tqdm(range(num_samples - 1), ncols=100):
        next_sample_idx = np.argmax(min_distances)
        labeled_idxs[next_sample_idx] = True
        select_samples_list.append(name_list[next_sample_idx])

        min_distances = np.minimum(min_distances, dist_mat[:, next_sample_idx])

    return select_samples_list

def generate_Coreset_plan(organ, feats_file, num_samples):
    """
    Generate a selection plan using Core-Set method.

    Parameters:
    - organ: str, name of the organ or dataset.
    - feats_file: str, path to the features file (.npz).
    - num_samples: int, the number of valuable samples to select.

    Saves:
    - A .npz file containing the selected sample paths.
    """
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_Coreset(feats=feats, name_list=name_list, num_samples=num_samples)
    paths = [f'./{organ}/data/{pid}.npz' for pid in select_samples_list]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Coreset_{num_samples}.npz'
    np.savez(save_path, paths=paths)

if __name__ == "main":
    generate_TypiClust_plan(organ='Heart', feats_file='./Heart/feats/Ours.npz', num_samples=5)