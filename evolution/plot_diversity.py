from pdb import set_trace as T
import json
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from evolution.diversity import calc_differential_entropy, calc_discrete_entropy_2, calc_convex_hull, calc_diversity_l2
from mpl_toolkits.mplot3d import axes3d, Axes3D

n_skills = 2
n_agents = 16
max_skill = 20000

def plot_div_2d(stats, title, path):
   ''' Plot a bunch of shit including histograms of diversity measures and reward over time'''
   fig, ax = plt.subplots()
   agent_skills = stats['skills']
   x = agent_skills[:, 0]
   y = agent_skills[:, 1]
#  ax = axs[0]
   plt.xlim(0, max_skill)
   plt.ylim(0, max_skill)
   plt.gca().set_aspect('equal', adjustable='box')
   plt.scatter(x, y)
   i = 0
#  for div_calc in DIV_CALCS:
#     if div_calc[0].__name__ == 'calc_differential_entropy':
#        infos = {}
#        score = div_calc[0](stats, verbose=False, infos=infos)
#        gaussian = infos['gaussian']
#     elif div_calc[0].__name__ == 'calc_convex_hull':
#        infos = {}
#        score = div_calc[0](stats, verbose=False, infos=infos)
#        hull = infos['hull']
#     elif div_calc[0].__name__ == 'calc_diversity_l2':
#        score = mean_pairwise = div_calc[0](stats, verbose=False)
#     else:
#        score = div_calc[0](stats, verbose=False)
#     div_str = '{}: {:4.1}'.format(div_calc[1], score)
#     print(div_str)
#     ax.text(2, i, div_str, horizontalalignment='right',
#        verticalalignment='center', transform=ax.transAxes)
#     i += 0.1
   plt.title(title)
#  plt.show()
   plt.savefig('{} scatter.png'.format(path))
   plt.close()
#  fig = figs[1]
#  ax = fig.add_subplot(111, projection='3d')
   fig, ax = plt.subplots()
   infos = {}
   score = calc_differential_entropy(stats, infos=infos)
   gaussian = infos['gaussian']
   plt.text(0.7, 0.7, '{:.2f}'.format(score), fontsize=20,transform=fig.transFigure)
   div_name = 'differential entropy'

#   ax = Axes3D(fig)
   ax.scatter(x, y, 5, 'blue')
##  x = np.linspace(0, 1000, max_skill)
##  y = np.linspace(0, 1000, max_skill)
##  X, Y = np.meshgrid(x, y)
   X, Y = np.mgrid[0:max_skill:100, 0:max_skill:100]
   pos = np.empty(X.shape + (2,))
   pos[:, :, 0] = X; pos[:, :, 1] = Y
##  xy = np.array(xy)
##  xy = np.moveaxis(xy, 0, -1)
   z = gaussian.pdf(pos)
   plt.contour(X, Y, z)
##  ax.plot_surface(X, Y, z)
#   ax.plot_wireframe(X, Y, z)
#  plt.title(title)
   plt.title('{}'.format(div_name))
   plt.savefig('{} normal.png'.format(path))
#  plt.show()
   plt.close()
   div_name = 'convex hull'
   infos = {}
   score = calc_convex_hull(stats, infos=infos)
   hull = infos['hull']
   plt.scatter(x, y)
   plt.xlim(0, max_skill)
   plt.ylim(0, max_skill)
   plt.text(0.7, 0.7, '{:.2}'.format(score), fontsize=20,transform=fig.transFigure)
   plt.title('{}'.format(div_name))
#  for simplex in hull.simplices:
#     plt.plot(x[simplex], y[simplex], 'k-')
   plt.fill(x[hull.vertices], y[hull.vertices], 'k', alpha=0.3)
#  plt.title(title)
   plt.savefig('{} {}.png'.format(path, div_name))
   plt.close()
   fig, ax = plt.subplots()
   div_name = 'pairwise distance'
   plt.text(0.7, 0.7, '{:.2}'.format(score), fontsize=20,transform=fig.transFigure)
   plt.title('{}'.format(div_name))
   score = calc_diversity_l2(stats)
   plt.scatter(x, y)
   for i in range(n_agents):
      for j in range(n_agents):
         plt.plot([agent_skills[i, 0], agent_skills[j, 0]], 
                  [agent_skills[i, 1], agent_skills[j, 1]]
                  )
#  plt.plot([0, mean_pairwise], [-1,-1], )
   plt.xlim(0, max_skill)
   plt.ylim(0, max_skill)
   plt.savefig('{} pairwise distance.png'.format(path))
   plt.close()
   fig, axs = plt.subplots(2)
   div_name = 'discrete entropy'
   score = calc_discrete_entropy_2(stats)
   ax = axs[0]
   plt.subplots_adjust(right=0.75)
   plt.text(0.8, 0.8, '{:.2f}'.format(score), fontsize=20,transform=fig.transFigure)
   plt.suptitle('{}'.format(div_name))
   x = np.arange(n_agents)
   width = 0.35
   rects1 = ax.bar(x - width/2, agent_skills.sum(axis=1))
   ax.set_ylabel('Mean experience')
   ax.set_xlabel('Agents')
   ax = axs[1]
   x = np.arange(n_skills)
   rects2 = ax.bar(x - width/2, agent_skills.sum(axis=0))
   ax.set_ylabel('Mean experience')
   ax.set_xlabel('Skills')
#  fig.tight_layout(pad=0)
   plt.savefig('{} discrete entropy.png'.format(path))
   plt.close()
   plt.figure()
   plt.title('agent-skill matrix {}'.format(path))
   im, cbar = heatmap(stats['skills'], {}, {})
   plt.savefig('{} agent-skill matrix'.format(path))
   plt.close()


def plot(agent_skills, title, path):
   agent_skills = np.clip(agent_skills, 0, max_skill)
   stats = {
         'skills': agent_skills,
         'lifespans': np.zeros((n_agents)) + 100,
         }
#  plot_div_2d(stats, axs[0])
   plot_div_2d(stats, title, path)

# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.
#
#     Parameters
#     ----------
#     data
#         A 2D numpy array of shape (N, M).
#     row_labels
#         A list or array of length N with the labels for the rows.
#     col_labels
#         A list or array of length M with the labels for the columns.
#     ax
#         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
#         not provided, use current axes or create a new one.  Optional.
#     cbar_kw
#         A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
#     cbarlabel
#         The label for the colorbar.  Optional.
#     **kwargs
#         All other arguments are forwarded to `imshow`.
#     """
#
#     if not ax:
#         ax = plt.gca()
#
#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)
#
#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#
#     # We want to show all ticks...
#     ax.set_xticks(np.arange(data.shape[1]))
#     ax.set_yticks(np.arange(data.shape[0]))
#     # ... and label them with the respective list entries.
#     ax.set_xticklabels(col_labels)
#     ax.set_yticklabels(row_labels)
#
#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#              rotation_mode="anchor")
#
#     # Turn spines off and create white grid.
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)
#
#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     return im, cbar



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
   """
   Create a heatmap from a numpy array and two lists of labels.

   Parameters
   ----------
   data
       A 2D numpy array of shape (N, M).
   row_labels
       A list or array of length N with the labels for the rows.
   col_labels
       A list or array of length M with the labels for the columns.
   ax
       A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
       not provided, use current axes or create a new one.  Optional.
   cbar_kw
       A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
   cbarlabel
       The label for the colorbar.  Optional.
   **kwargs
       All other arguments are forwarded to `imshow`.
   """

   if not ax:
      ax = plt.gca()

   # Plot the heatmap
   im = ax.imshow(data, **kwargs)

   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
   cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

   # We want to show all ticks...
   ax.set_xticks(np.arange(data.shape[1]))
   ax.set_yticks(np.arange(data.shape[0]))
   # ... and label them with the respective list entries.
   ax.set_xticklabels(col_labels)
   ax.set_yticklabels(row_labels)

   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False,
                  labeltop=True, labelbottom=False)

   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")

   # Turn spines off and create white grid.
   #ax.spines[:].set_visible(False)

   ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
   ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
   ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
   ax.tick_params(which="minor", bottom=False, left=False)

   return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
   """
   A function to annotate a heatmap.

   Parameters
   ----------
   im
       The AxesImage to be labeled.
   data
       Data used to annotate.  If None, the image's data is used.  Optional.
   valfmt
       The format of the annotations inside the heatmap.  This should either
       use the string format method, e.g. "$ {x:.2f}", or be a
       `matplotlib.ticker.Formatter`.  Optional.
   textcolors
       A pair of colors.  The first is used for values below a threshold,
       the second for those above.  Optional.
   threshold
       Value in data units according to which the colors from textcolors are
       applied.  If None (the default) uses the middle of the colormap as
       separation.  Optional.
   **kwargs
       All other arguments are forwarded to each call to `text` used to create
       the text labels.
   """

   if not isinstance(data, (list, np.ndarray)):
      data = im.get_array()

   # Normalize the threshold to the images color range.
   if threshold is not None:
      threshold = im.norm(threshold)
   else:
      threshold = im.norm(data.max()) / 2.

   # Set default alignment to center, but allow it to be
   # overwritten by textkw.
   kw = dict(horizontalalignment="center",
             verticalalignment="center")
   kw.update(textkw)

   # Get the formatter in case a string is supplied
   if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

   # Loop over the data and create a `Text` for each "pixel".
   # Change the text's color depending on the data.
   texts = []
   for i in range(data.shape[0]):
      for j in range(data.shape[1]):
         kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
         text = im.axes.text(j, i, valfmt(data[i, j], pos=(i, j)), **kw)
         texts.append(text)

   return texts

exp_colors = {
      'cppn': 'purple',
      'pattern': 'green',
      'random': 'blue',
      'current': 'black',
      }

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--compare_models', nargs='+', default=None)
   parser.add_argument('--map', default='PCG')
   parser.add_argument('--infer_idx', default=0)
   args = parser.parse_args()
   if args.compare_models is None:
   #  plt.rcParams["figure.figsize"] = (20,3)
   #  figs, axs = plt.subplots(3, 3)
      title = 'uniform random population'
      path = os.path.join('../eval_experiment', 'toy_cases', title)
      agent_skills = np.random.randint(0, 20000, (16, 2))
      plot(agent_skills, title, path)
   #  plt.show()

      title = 'uniform generalists'
      path = os.path.join('../eval_experiment', 'toy_cases', title)
      mean = (10000, 10000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills = np.random.multivariate_normal(mean, cov, n_agents)
   #  a_skill = np.zeros((1, n_skills)) + 10000
   #  agent_skills = np.repeat(a_skill, n_agents, axis=0)
      plot(agent_skills, title, path)
   #  plt.show()

      title = 'uniform specialists'
      path = os.path.join('../eval_experiment', 'toy_cases', title)
      mean = (18000, 2000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills = np.random.multivariate_normal(mean, cov, n_agents)
   #  a_skill = np.array([[5000, 15000]])
   #  agent_skills = np.repeat(a_skill, n_agents, axis=0)
      plot(agent_skills, title, path)
   #  plt.show()

      title = 'bimodal specialists'
      path = os.path.join('../eval_experiment', 'toy_cases', title)
      mean = (18000, 2000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills_1 = np.random.multivariate_normal(mean, cov, n_agents//2)
      mean = (2000, 18000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills_2 = np.random.multivariate_normal(mean, cov, n_agents//2)
   #  a_skill_1 = np.array([[1000, 19000]])
   #  a_skill_2 = np.array([[19000, 1000]])
   #  agent_skills_1 = np.repeat(a_skill_1, n_agents//2, axis=0)
   #  agent_skills_2 = np.repeat(a_skill_2, n_agents//2, axis=0)
      agent_skills = np.vstack((agent_skills_1, agent_skills_2))
      plot(agent_skills, title, path)

      title = 'trimodal specialists'
      path = os.path.join('../eval_experiment', 'toy_cases', title)
      mean = (2000, 2000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills_1 = np.random.multivariate_normal(mean, cov, n_agents//3)
      mean = (18000, 2000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills_2 = np.random.multivariate_normal(mean, cov, n_agents//3)
      mean = (2000, 18000)
      cov = [[1e6, 1e5],[1e5,1e6]]
      agent_skills_3 = np.random.multivariate_normal(mean, cov, n_agents-2*(n_agents//3))
   #  a_skill_1 = np.array([[1000, 19000]])
   #  a_skill_2 = np.array([[19000, 1000]])
   #  agent_skills_1 = np.repeat(a_skill_1, n_agents//2, axis=0)
   #  agent_skills_2 = np.repeat(a_skill_2, n_agents//2, axis=0)
      agent_skills = np.vstack((agent_skills_1, agent_skills_2, agent_skills_3))
      plot(agent_skills, title, path)
   else:
      map_dir = os.path.join('../eval_experiment', args.map, str(args.infer_idx))
      model_stats = {}
      for model_name in args.compare_models:
         model_dir = os.path.join(map_dir, model_name)
         model_dir = os.path.abspath(os.path.join(model_dir, 'stats.json'))
         with open(model_dir, 'r') as f:
            stats = json.load(f)
            model_stats[model_name] = stats

      width = 0.35
#     figs, axs = plt.subplots(len(stats))
      plt.figure(figsize=(10,10))
#     plt.suptitle('Model performance on map: {}'.format(map_dir))
      for i, metric_name in enumerate(stats.keys()):
         ax = plt.subplot2grid((len(stats), 1), (i, 0))
         scores = [m_stats[metric_name] for m_stats in model_stats.values()]
   #     scores = [score for score in stats.values()]
         means = [score['mean'] if isinstance(score, dict) else score for score in scores]
#        color = ['black' if 'current' in model_name else 'blue' if 'random' in model_name else 'green' for model_name in model_stats.keys()]
         color = []
         for model_name in model_stats.keys():
            has_color = False
            if model_name == args.map:
               color.append('red')
               continue
            for exp_kw in exp_colors:
               if exp_kw in model_name:
                  color.append(exp_colors[exp_kw])
                  has_color = True
            if not has_color:
               color.append('gray')
         print(color)
    #    stds = [score['std'] for score in scores]
   #     x = np.arange(len(means))
         x = np.arange(len(args.compare_models))
         ax.bar(x, means, color=color)
         ax.set_title(metric_name)
      print(args.compare_models)
      print(x)
      ax.set_xticks(x)
      ax.set_xticklabels(args.compare_models, rotation=45, ha='right')
      plt.subplots_adjust(top=0.8)
      plt.tight_layout()
      plt.savefig(os.path.join(map_dir, 'compare_models_MAP_' + args.map + '_' + '_'.join(args.compare_models) +'.png'))
