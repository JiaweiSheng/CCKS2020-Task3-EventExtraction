import pandas as pd
import numpy as np


def load_results_csv(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    keys = lines[0].strip().split(',')

    results_dict = {}
    for line in lines[1:]:
        spts = line.strip().split(',')
        results_dict[spts[0]] = [int(item) for item in spts[1:]]
    return keys, results_dict


def vote_for_final(results_models: list):
    ids = results_models[0].keys()
    num_class = len(results_models[0][list(ids)[0]])
    results_final = {}
    for idx in ids:
        scores_final = [0] * num_class
        for results_dict in results_models:
            scores = results_dict[idx]
            for i in range(num_class):
                scores_final[i] += scores[i]
        results_final[idx] = scores_final
    return results_final


def write(fn, ids, result, label_list):
    ids = [[item] for item in ids]
    ids = np.array(ids)
    df1 = pd.DataFrame(ids, index=None)
    df2 = pd.DataFrame(result, index=None)
    all_df = pd.concat([df1, df2], axis=1)

    all_df.columns = ['id'] + label_list
    all_df.to_csv(fn, index=False)


if __name__ == '__main__':
    results_models = []
    for i in range(10):
        fn = f'results/sig/spt{str(i + 1)}.trans.cls'
        keys, results_dict = load_results_csv(fn)
        results_models.append(results_dict)

    results_final = vote_for_final(results_models)
    ids = results_final.keys()
    result = results_final.values()
    write('results/essemble/trans_5.cls', ids, result, keys[1:])
