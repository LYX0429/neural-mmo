from pdb import set_trace as T
import numpy as np
import scipy
from scipy.spatial import ConvexHull

import skbio


def diversity_calc(config):
   if config.FITNESS_METRIC == 'L2':
      calc_diversity = calc_diversity_l2
   elif config.FITNESS_METRIC == 'Differential':
      calc_diversity = calc_differential_entropy
   elif config.FITNESS_METRIC == 'Discrete':
      calc_diversity = calc_discrete_entropy_2
   elif config.FITNESS_METRIC == 'Hull':
       calc_diversity = calc_convex_hull
   elif config.FITNESS_METRIC == 'Sum':
       calc_diversity = sum_experience
   else:
       raise Exception('Unsupported fitness function: {}'.format(self.config.FITNESS_METRIC))
   return calc_diversity

   def sum_experience(agent_stats, skill_headers=None, verbose=False):
      # No need to weight by lifespan.
      agent_skills = agent_stats['skills']
      lifespans = agent_stats['lifespans']
      a_skills = np.vstack(agent_skills)
      a_lifespans = np.hstack(lifespans)
   #  weights = sigmoid_lifespan(a_lifespans)
      n_agents, n_skills = a_skills.shape
   #  avg_skill = (weights * a_skills).sum() / (n_agents * n_skills)
      mean_xp = a_skills.sum() / (n_agents * n_skills)
      if verbose:
         print('skills')
         print(a_skills.T)
         print('lifespans')
         print(a_lifespans)
         print('mean xp:', mean_xp)
         print()

      return mean_xp

def sigmoid_lifespan(x):
   res = 1 / (1 + np.exp(0.1*(-x+50)))
#  res = scipy.special.softmax(res)

   return res

def calc_differential_entropy(agent_stats, skill_headers=None, verbose=False, infos={}):
   # Penalize if under max pop agents living
  #for i, a_skill in enumerate(agent_skills):
  #   if a_skill.shape[0] < max_pop:
  #      a = np.mean(a_skill, axis=0)
  #      a_skill = np.vstack(np.array([a_skill] + [a for _ in range(max_pop - a_skill.shape[0])]))
  #      agent_skills[i] = a_skill
   # if there are stats from multiple simulations, we consider agents from all simulations together
   #FIXME: why
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']

   assert len(agent_skills) == len(lifespans)
   a_skills = np.vstack(agent_skills)
   a_lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(a_lifespans)
#  assert len(agent_skills) == 1
  #a_skills = agent_skills[0]
   # FIXME: Only applies to exploration-only experiment
  #print('exploration')
  #print(a_skills.transpose()[0])
   if verbose:
      print(skill_headers)
      print(a_skills.transpose())
      print(len(agent_skills), 'populations')
      print('lifespans')
      print(a_lifespans)

  #if len(a_lifespans) == 1:
  #   score = 0
  #else:
   mean = np.average(a_skills, axis=0, weights=weights)
   cov = np.cov(a_skills,rowvar=0, aweights=weights)
#     cov = np.array([[2,1],[1,2]])
#    T()
   gaussian = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
   infos['gaussian'] = gaussian
   score = gaussian.entropy()

#  print(np.array(a_skills))
   if verbose:
      print('score:', score)

   return score


def calc_convex_hull(agent_stats, skill_headers=None, verbose=False, infos={}):
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']
   agent_skills = np.vstack(agent_skills)
   n_skills = agent_skills.shape[1]

   lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(lifespans)
   if verbose:
      print('skills:')
      print(agent_skills.transpose())
      print('lifespans:')
      print(lifespans)
      print(len(agent_stats['lifespans']), 'populations')
   n_agents = lifespans.shape[0]
   mean_agent = agent_skills.mean(axis=0)
   mean_agents = np.repeat(mean_agent.reshape(1, mean_agent.shape[0]), n_agents, axis=0)
   agent_deltas = agent_skills - mean_agents
   agent_skills = mean_agents + (weights * agent_deltas.T).T
   if n_skills == 1:
      # Max distance, i.e. a 1D hull
      score = agent_skills.max() - agent_skills.mean()
   else:
      try:
          hull = ConvexHull(agent_skills, qhull_options='QJ')
          infos['hull'] = hull
          score = hull.volume
          score = score ** (1 / n_skills)
      except Exception as e:
          print(e)
          score = 0
      if verbose:
         print('score:', score)

   return score

def calc_discrete_entropy_2(agent_stats, skill_headers=None, verbose=False):
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']
   agent_skills_0 = agent_skills= np.vstack(agent_skills)
   lifespans = np.hstack(lifespans)
   n_agents = lifespans.shape[0]
   if n_agents == 1:
       return -np.float('inf')
   n_skills = agent_skills.shape[1]
   if verbose:
       print('skills')
       print(agent_skills_0.transpose())
       print('lifespans')
       print(lifespans)
   weights = sigmoid_lifespan(lifespans)
   agent_skills_1 = agent_skills_0.transpose()
   # discretize
   agent_skills = np.where(agent_skills==0, 0.0000001, agent_skills)
   # contract population toward mean according to lifespan
   # mean experience level for each agent
   mean_skill = agent_skills.mean(axis=1)
   # mean skill vector of an agent
   mean_agent = agent_skills.mean(axis=0)
   assert mean_skill.shape[0] == n_agents
   assert mean_agent.shape[0] == n_skills
   mean_skills = np.repeat(mean_skill.reshape(mean_skill.shape[0], 1), n_skills, axis=1)
   mean_agents = np.repeat(mean_agent.reshape(1, mean_agent.shape[0]), n_agents, axis=0)
   agent_deltas = agent_skills - mean_agents
   skill_deltas = agent_skills - mean_skills
   a_skills_skills = mean_agents + (weights * agent_deltas.transpose()).transpose()
   a_skills_agents = mean_skills + (weights * skill_deltas.transpose()).transpose()
   div_agents = skbio.diversity.alpha_diversity('shannon', a_skills_agents).mean()
   div_skills = skbio.diversity.alpha_diversity('shannon', a_skills_skills.transpose()).mean()
 # div_lifespans = skbio.diversity.alpha_diversity('shannon', lifespans)
   score = -(div_agents * div_skills)#/ div_lifespans#/ len(agent_skills)**2
   score = score#* 100  #/ (n_agents * n_skills)
   if verbose:
       print('Score:', score)

   return score


def calc_discrete_entropy(agent_stats, skill_headers=None):
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']
   agent_skills_0 = np.vstack(agent_skills)
   agent_lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(agent_lifespans)
   agent_skills = agent_skills_0.transpose() * weights
   agent_skills = agent_skills.transpose()
   BASE_VAL = 0.0001
   # split between skill and agent entropy
   n_skills = len(agent_skills[0])
   n_pop = len(agent_skills)
   agent_sums = [sum(skills) for skills in agent_skills]
   i = 0

   # ensure that we will not be dividing by zero when computing probabilities

   for a in agent_sums:
       if a == 0:
           agent_sums[i] = BASE_VAL * n_skills
       i += 1
   skill_sums = [0 for i in range(n_skills)]

   for i in range(n_skills):

       for a_skills in agent_skills:
           skill_sums[i] += a_skills[i]

       if skill_sums[i] == 0:
           skill_sums[i] = BASE_VAL * n_pop

   skill_ents = []

   for i in range(n_skills):
       skill_ent = 0

       for j in range(n_pop):

           a_skill = agent_skills[j][i]

           if a_skill == 0:
               a_skill = BASE_VAL
           p = a_skill / skill_sums[i]

           if p == 0:
               skill_ent += 0
           else:
               skill_ent += p * np.log(p)
       skill_ent = skill_ent / (n_pop)
       skill_ents.append(skill_ent)

   agent_ents = []

   for j in range(n_pop):
       agent_ent = 0

       for i in range(n_skills):

           a_skill = agent_skills[j][i]

           if a_skill == 0:
               a_skill = BASE_VAL
           p = a_skill / agent_sums[j]

           if p == 0:
               agent_ent += 0
           else:
               agent_ent += p * np.log(p)
       agent_ent = agent_ent / (n_skills)
       agent_ents.append(agent_ent)
   agent_score =  np.mean(agent_ents)
   skill_score =  np.mean(skill_ents)
#  score = (alpha * skill_score + (1 - alpha) * agent_score)
   score = -(skill_score * agent_score)
   score = score * 100#/ n_pop**2
   print('agent skills:\n{}\n{}'.format(skill_headers, np.array(agent_skills_0.transpose())))
   print('lifespans:\n{}'.format(lifespans))
#  print('skill_ents:\n{}\nskill_mean:\n{}\nagent_ents:\n{}\nagent_mean:{}\nscore:\n{}\n'.format(
#      np.array(skill_ents), skill_score, np.array(agent_ents), agent_score, score))
   print('score:\n{}'.format(score))

   return score


def calc_diversity_l2(agent_stats, skill_headers=None, verbose=False):
   if 'skills' not in agent_stats:
      return 0
   agent_skills = agent_stats['skills']
   lifespans = agent_stats['lifespans']
   assert len(agent_skills) == len(lifespans)
   a_skills = np.vstack(agent_skills)
   a_lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(a_lifespans)
   weight_mat = np.outer(weights, weights)
  #assert len(agent_skills) == 1
   a = a_skills
   n_agents = a.shape[0]
   b = a.reshape(n_agents, 1, a.shape[1])
   # https://stackoverflow.com/questions/43367001/how-to-calculate-euclidean-distance-between-pair-of-rows-of-a-numpy-array
   distances = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
   w_dists = weight_mat * distances
   score = np.sum(w_dists)/2
   score = score / n_agents

   if verbose:
#  print(skill_headers)
       print('agent skills:\n{}'.format(a.transpose()))
       print('lifespans:\n{}'.format(a_lifespans))
       print('score:\n{}\n'.format(
       score))

   return score

DIV_CALCS = [(calc_diversity_l2, 'mean pairwise L2'), (calc_differential_entropy, 'differential entropy'), (calc_discrete_entropy_2, 'discrete entropy'), (calc_convex_hull, 'convex hull volume')]
