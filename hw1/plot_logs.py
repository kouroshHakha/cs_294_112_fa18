"""
>>> python plot_logs.py q2/3 log/Humanoid-v2.pkl
>>> python plot_logs.py q3/2 log/Humanoid-v2.pkl --dagger_file_loc log/Humanoid_dagger.pkl
"""
import numpy as np
import pickle
import os
import matplotlib . pyplot as plt
import seaborn as sns


def plot_q2_3(loc):
    assert os.path.exists(loc), 'path {} doesn\'t exist'.format(loc)

    with open(loc, 'rb') as rf:
        data = pickle.load(rf)

    for key in ['hyperparams', 'returns_list']:
        assert key in data.keys(), '{} is not a key in the file {}'.format(key, loc)

    hyper_params = np.array(data['hyperparams'])
    returns_list = np.array(data['returns_list']).transpose()
    sns.tsplot(time=hyper_params, data=returns_list)
    plt.title('Humanoid-v2: BC performance vs. number of training epochs')
    plt.ylabel('Return dist. across 20 rollouts')
    plt.xlabel('number of epochs')
    plt.show()

def plot_q3_2(args):
    assert os.path.exists(args.bc_file_loc), 'path {} doesn\'t exist'.format(args.bc_file_loc)
    assert os.path.exists(args.dagger_file_loc), 'path {} doesn\'t exist'.format(args.dagger_file_loc)

    with open(args.bc_file_loc, 'rb') as rf:
        data_bc = pickle.load(rf)
    with open(args.dagger_file_loc, 'rb') as rf:
        data_dagger = pickle.load(rf)

    for key in ['hyperparams', 'returns_list']:
        assert key in data_dagger.keys(), '{} is not a key in the file {}'.format(key, args.dagger_file_loc)
        assert key in data_bc.keys(), '{} is not a key in the file {}'.format(key, args.bc_file_loc)


    dagger_iter = np.arange(0, len(data_dagger['hyperparams']), step=1)
    dagger_returns_list = np.array(data_dagger['returns_list']).transpose()
    bc_returns = np.array(data_bc['returns_list'])[-1:]
    bc_curve = np.repeat(bc_returns, len(dagger_iter), axis=0).transpose()

    sns.tsplot(time=dagger_iter, data=dagger_returns_list)
    sns.tsplot(time=dagger_iter, data=bc_curve, color='r')

    plt.title('Humanoid-v2: Comparison of 20 rollouts of dagger with iter=5, epoch=10 per iter and bc with epoch=50 ')
    plt.ylabel('Return dist. across 20 rollouts')
    plt.xlabel('dagger iteration')
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('what', type=str, help='for q2.3 put q2/3')
    parser.add_argument('bc_file_loc', type=str, help='the location of the file containing plotting information')
    parser.add_argument('--dagger_file_loc', type=str, help='the location of the file containing plotting information')
    args = parser.parse_args()

    if args.what == 'q2/3':
        plot_q2_3(args.bc_file_loc)
    elif args.what == 'q3/2':
        plot_q3_2(args)

if __name__ == '__main__':
    main()