import numpy as np
import os.path as osp
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

def read_tsv(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            scores.append(float(line.strip()))
    return np.array(scores)

def construct_kernels(X):
    gamma_values = [0.1, 1, 10]
    kernels = []
    for gamma in gamma_values:
        pairwise_dist = pairwise_distances(X, metric='euclidean')
        kernel = np.exp(-gamma * pairwise_dist**2 / 2)
        kernels.append(kernel)
    return kernels

def multiple_kernel_kmeans(kernels, k, max_iter=100):
    n_samples = kernels[0].shape[0]
    cluster_assignments = np.random.choice(k, n_samples)
    
    for iteration in range(max_iter):
        combined_kernel = np.zeros((n_samples, n_samples))
        for m in range(len(kernels)):
            combined_kernel += kernels[m]
        
        new_assignments = np.zeros(n_samples, dtype=int)
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
        
        # Check if any cluster has no samples, reinitialize those clusters
        for j in range(k):
            if np.sum(new_assignments == j) == 0:
                new_assignments[np.random.choice(n_samples)] = j

        if np.array_equal(cluster_assignments, new_assignments):
            break
        
        cluster_assignments = new_assignments
    
    centers = np.zeros((k, n_samples))
    for j in range(k):
        in_cluster = (cluster_assignments == j)
        if np.sum(in_cluster) == 0:
            continue
        cluster_indices = np.where(in_cluster)[0]
        cluster_points = combined_kernel[cluster_indices][:, cluster_indices]
        centers[j, cluster_indices] = np.mean(cluster_points, axis=0)
    
    return centers, cluster_assignments

def select_valuable_samples_Kmeans(feats, name_list, num_samples):
    kernels = construct_kernels(feats)
    
    centers, labels = multiple_kernel_kmeans(kernels, num_samples)
    
    selected_indices = []
    for i in range(num_samples):
        cluster_points = feats[labels == i]
        cluster_names = name_list[labels == i]

        if len(cluster_points) == 0:
            continue

        cluster_center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        closest_index = np.argmin(distances)
        
        selected_indices.append(np.where(name_list == cluster_names[closest_index])[0][0])

    return np.array(selected_indices)

def generate_Kmeans_plan(organ, feats_file, score_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])

    select_samples_list = select_valuable_samples_Kmeans(feats=feats, name_list=name_list, num_samples=num_samples)

    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in name_list[select_samples_list]]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Ours_without_searching_{num_samples}.npz'
    np.savez(save_path, paths=paths)

generate_Kmeans_plan(organ='Lung', feats_file='./Lung/feats/Ours.npz', score_file='./Lung/feats/Ours_scores.tsv', num_samples=5)
