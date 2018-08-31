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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('what', type=str, help='for q2.3 put q2/3')
    parser.add_argument('loc', type=str, help='the location of the file containing plotting information')
    args = parser.parse_args()

    if args.what == 'q2/3':
        plot_q2_3(args.loc)

if __name__ == '__main__':
    main()