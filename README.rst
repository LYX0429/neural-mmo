*****************
Readme: joint generator-player optimization in multi-agent environments
*****************

Overview & current progress
#####

This is an ongoing project that builds on Neural MMO, an open-ended, artificial life-like environment for multi-agent reinforcement learning (original Readme below). It seeks to implement a training loop that jointly optimizes player(s) and map-generator(s). At present, it uses the original NMMO setup to train RL policies to play as selfish agents (with negative reward for dying, usually learning to forage for food and water, and sometimes fight one another), while simultaneously evolving a population of maps using the MAP-Elites quality diversity algorithm. (We plan to supplant direct optimization of the maps with optimization of map-generators in the near future, where these generators can be implemented as neural networks, like NCAs or generative CPPNs, or similar.) 

While agents are always trained to survive, maps can be optimized to induce certain behavior in the players during learning, including:

* Maximum lifespan (resulting in trivially easy/prosperous maps)
* Maximum diversity in skill-space (resulting in a variety of niches demanding specialized behaviors)

The following map-generator objectives are in development or not yet implemented:

* Maximum homogeneity among agents (implemented, untested)
* Maximum discrepancy of success between two distinct, learning player-policies (implemented, untested)
* Maximum learning progress among players after several updates (implemented, likely requires more learning on each map, potentially computationally infeasible)
* Mixed success among agents, indicative of maps that are neither too hard nor too easy for the current policy (not implemented)

Installation
#######

Create a conda environment and install dependencies:
::
  conda create -n nmmo python==3.8
  conda activate nmmo
  bash scripts/setup.sh

Instructions
####

The file evo_batch.py runs batches of experiments by iterating through hyperparameters and calling ForgeEvo.py, either sequentially on a local machine, or by queueing batches of parallel experiments on a SLURM cluster. You can run it with:
::
  python evo_batch.py --local
  
(dropping the --local argument if you're on a cluster).

If you are attempting to use a GPU (recommended) and you encounter an IndexError in ray/rllib/policy/torch_policy.py when attempting to set self.device, replace the lign assigning gpu_ids in this file with:
::
  gpu_ids = [0]

To determine what batch of experiments will be run, (un)comment the appropriate hyperparameters listed in evo_batch.py. We discuss these hyperparamters below.

genomes
********************

How map-generators are represented. Each genome defines an individual that implemente gen_map() and mutate(). At the beginning (and/or throughout generator optimization), the genome is initialized randomly, corresponding to some random map, then cloned and mutated, with each mutation (generally) leading to some change in the map produced by the individual's map-generation function. 

* Simplex
* NCA
* CPPN
* Primitives
* LSystem
* TileFlip
* All

generator objectives
*********************

The objective that map-generators seek to maximize during optimization. 

************************
Readme: Neural MMO
************************

.. |icon| image:: docs/source/resource/icon/icon_pixel.png

.. figure:: docs/source/resource/image/splash.png


|icon| Welcome to the Platform!
###############################

Note (Feb 12): We are in the middle of launch. v1.5 should be up by some time on Monday. Use the v1.4 branch until then.

`[Demo Video] <https://youtu.be/y_f77u9vlLQ>`_ | `[Discord] <https://discord.gg/BkMmFUC>`_ | `[Twitter] <https://twitter.com/jsuarez5341>`_

Neural MMO is a massively multiagent AI research environment inspired by Massively Multiplayer Online (MMO) role playing games. The project is under active development with major updates every 3-6 months. This README is a stub -- all of our `[Documentation] <https://jsuarez5341.github.io>`_ is hosted by github.io.
