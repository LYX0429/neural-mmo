import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np

def plot_exp(exp_name, render=False):

    my_dpi = 96
    log_path = 'evo_experiment/{}/log.csv'.format(exp_name)
    im_path = 'evo_experiment/{}/fitness.png'.format(exp_name)

    with open(log_path, mode='r') as log_file:
        results = {}
        csv_reader = csv.DictReader(log_file)
        curr_epoch = 0
        newest_epoch = 0
        # NB: we've written to the csv incorrectly :( annoying

        for row in csv_reader:
            row = list(row.values())

            if row[0].startswith('epoch'):
                curr_epoch = int(row[0].split(' ')[-1])
            elif row[0] == 'id':
                pass
            else:
                if curr_epoch not in results:
                    results[curr_epoch] = {}
                map_id = row[0]
                results[curr_epoch][map_id] = row[1]
    ep_scores = []
    i = 0

    while i in results:
        ep_scores.append([])
        ep_results = results[i]

        for map_id, id_results in ep_results.items():
            score = float(id_results[0])
            ep_scores[i].append(score)
        i += 1
    avg_scores = []

    for scores in ep_scores:
        scores = np.array(scores)
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)
    x = [i for i in range(len(ep_scores))]
    ind_x = [i for i in range(len(ep_scores)) for individual in range(len(scores))]
    colors = ['darkgreen', 'm', 'g', 'y', 'salmon', 'darkmagenta', 'orchid', 'darkolivegreen', 'mediumaquamarine',
            'mediumturquoise', 'cadetblue', 'slategrey', 'darkblue', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet',
            'fuchsia', 'deeppink', 'olive', 'orange', 'maroon', 'lightcoral', 'firebrick', 'black', 'dimgrey', 'tomato',
            'saddlebrown', 'greenyellow', 'limegreen', 'turquoise', 'midnightblue', 'darkkhaki', 'darkseagreen', 'teal',
            'cyan', 'lightsalmon', 'springgreen', 'mediumblue', 'dodgerblue', 'mediumpurple', 'darkslategray', 'goldenrod',
            'indigo', 'steelblue', 'coral', 'mistyrose', 'indianred']
#   fig, ax = plt.subplots(figsize=(1800/my_dpi, 900/my_dpi), dpi=my_dpi)
    fig, ax = plt.subplots()
    if True:
        for i in range(len(scores)):
            ind_y = []
            for e_scores in ep_scores:
                if len(e_scores) > i:
                    ind_y.append(e_scores[-i])
                else:
                    ind_y.append(0)
            ax.scatter(x, ind_y, s=0.1, c=colors[i%48])
    avg_scores = [np.mean(np.array(scores)) for scores in ep_scores]
    std = [np.std(np.array(scores)) for scores in ep_scores]
    ax.plot(avg_scores, c='indigo')
    markers, caps, bars = ax.errorbar(x, avg_scores, yerr=std,
                                       ecolor='purple')
    [bar.set_alpha(0.03) for bar in bars]
    plt.ylabel('fitness')
    plt.xlabel('generation')
    plt.title('evolved diversity')
    plt.savefig(im_path, dpi=my_dpi)
    if render:
       plt.show()
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name',
                        default='scratch',
                        help='name of the experiment')
    args = parser.parse_args()

    #exp_name = 'entropy'
    #exp_name = 'special_and_alive'
    #exp_name = 'skill_entropy'
    #exp_name = 'skill_entropy_life'
    #exp_name = 'entropy_2'
    #exp_name = 'skill_evolver'
    #exp_name = 'scratch'

    exp_name = args.experiment_name
    plot_exp(exp_name, render=True)
