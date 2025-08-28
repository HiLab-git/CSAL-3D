import numpy as np
import os.path as osp
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

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

def calculate_typicality(cluster_points):
    distances = cosine_distances(cluster_points)
    typicality = 1 / (np.mean(distances, axis=1) + 1e-10)
    return typicality

def select_valuable_samples_Uncertainty_Kmeans(feats, score, name_list, num_samples, search_nums):
    scaler = MinMaxScaler()
    uncertainties = scaler.fit_transform(score.reshape(-1, 1)).flatten()
    
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


if __name__ == '__main__'
    generate_Uncertainty_Kmeans_plan(organ='BrainTumour_Multi-Modality', feats_file='./BrainTumour/feats/Ours.npz', score_file='./BrainTumour/feats/Ours_scores.tsv', num_samples=30, search_nums=3)
