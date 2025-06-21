import numpy as np
import os.path as osp

def read_tsv(file_path):
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            scores.append(float(line.strip()))
    return np.array(scores)

def select_valuable_samples(scores, num_samples):
    selected_indices = np.argsort(scores)[-num_samples:]
    return selected_indices

def generate_selection_plan(organ, feats_file, score_file, num_samples):
    np.random.seed(0)
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    name_list = np.array(data_dict['name_list'])

    scores = read_tsv(score_file)

    select_samples_list = select_valuable_samples(scores=scores, num_samples=num_samples)

    paths = [osp.join(f'./{organ}/data/{pid}.npz') for pid in name_list[select_samples_list]]

    print(f'Select:\n{paths}')
    save_path = f'./{organ}/plans/Ours_uncertainty_{num_samples}.npz'
    np.savez(save_path, paths=paths)

generate_selection_plan(organ='Lung', feats_file='./Lung/feats/Ours.npz', score_file='./Lung/feats/Ours_scores.tsv', num_samples=5)
