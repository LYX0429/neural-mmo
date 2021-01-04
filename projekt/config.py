from pdb import set_trace as T
from forge.blade import core
import os

class Config(core.Config):
   EVO_MAP = False
   MELEE_MULT = 45 / 99
   RANGE_MULT = 32 / 99
   MAGE_MULT =  24 / 99
   # Model to load. None will train from scratch
   # Baselines: recurrent, attentional, convolutional
   # "current" will resume training custom models

   MODEL        = 'current'
   SCRIPTED_BFS = False
   SCRIPTED_DP  = False
   EVALUATE     = False
   LOCAL_MODE   = False

   # Model dimensions
   EMBED  = 64
   HIDDEN = 64

   # Environment parameters
   NPOP = 1    # Number of populations
   NENT = 1024 # Maximum population size
   NMOB = 1024 # Number of NPCS

   NMAPS = 256 # Number maps to generate

   # Evaluation parameters
   EVALUATION_HORIZON = 2048

   #Agent vision range
   STIM    = 7

   #Maximum number of observed agents
   N_AGENT_OBS = 100

   # Whether to share weights across policies
   # The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = True
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP

   # Evaluation
   LOG_DIR = 'experiment/'
   LOG_FILE = 'evaluation.npy'
   LOG_FIGURE = 'evaluation.html'

   # Visualization
   THEME_DIR = 'forge/blade/systems/visualizer/'
   THEME_NAME = 'web'  # publication or web
   THEME_FILE = 'theme_temp.json'
   THEME_WEB_INDEX = 'index_web.html'
   THEME_PUBLICATION_INDEX = 'index_publication.html'
   PORT = 5006
   PLOT_WIDTH = 1920
   PLOT_HEIGHT = 270
   PLOT_COLUMNS = 4
   PLOT_TOOLS = False
   PLOT_INTERACTIVE = False

#Small map preset
class SmallMap(Config):
   MODEL                   = 'small-map'

   NENT                    = 128
   NMOB                    = 32

   TERRAIN_MODE            = 'contract'
   TERRAIN_LERP            = False

   TERRAIN_SIZE            = 80 
   TERRAIN_OCTAVES         = 1
   TERRAIN_FOREST_LOW      = 0.30
   TERRAIN_FOREST_HIGH     = 0.75
   TERRAIN_GRASS           = 0.715
   TERRAIN_ALPHA           = -0.025
   TERRAIN_BETA            = 0.035

   TERRAIN_DIR             = Config.TERRAIN_DIR_SMALL
   ROOT                    = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')

   INVERT_WILDERNESS       = True
   WILDERNESS              = False

   NPC_LEVEL_MAX           = 35
   NPC_LEVEL_SPREAD        = 5
   NPC_SPAWN_PASSIVE       = 0.00
   NPC_SPAWN_NEUTRAL       = 0.60
   NPC_SPAWN_AGGRESSIVE    = 0.80

class ResourcesTest(SmallMap):
   NENT                 = 8
   NMOB                 = 0
   MODEL                = 'current'
   TERRAIN_SIZE         = 80
   TERRAIN_DIR          = Config.TERRAIN_DIR_SMALL
   ROOT                 = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')
#  TERRAIN_RENDER       = True
   TERRAIN_ALPHA = 0
   TERRAIN_BETA = 0
   TERRAIN_LAVA         = 0.0
   TERRAIN_WATER        = 0.25
   TERRAIN_FOREST_LOW   = 0.35
   TERRAIN_GRASS        = 0.7
   TERRAIN_FOREST_HIGH  = 0.75
   TERRAIN_TREE         = 0.8
   TERRAIN_OREROCK      = 0.9

class EvoNMMO(ResourcesTest):

   INFER_IDX = 3
 # INFER_IDX = 79766
 # INFER_IDX = 80117
   # How to measure diversity of agents on generated map.
   FITNESS_METRIC = 'L2' # 'Differential', 'L2', 'Discrete'
   GENE = 'Random'  # CPPN, Pattern, Random
   THRESHOLD = False
   TERRAIN_MODE = 'contract'
   EVO_MAP = True
   RENDER = False
   MODEL = 'current'
   NENT = 8  # Maximum population size
   TERRAIN_SIZE = 64
   EVO_DIR = 'all_random_l2'
#  ROOT = os.path.join(os.getcwd(), 'evo_experiment', EVO_DIR, 'maps', 'map')
   N_EVO_MAPS = 48
   MAX_STEPS = 100
   MATURE_AGE = 3
   ROLLING_FITNESS = 25  # Size of window to use while calculating mean rolling fitness
   TERRAIN_RENDER = True
   TERRAIN_WATER        = 0.15
   TERRAIN_GRASS        = 0.35
   TERRAIN_LAVA         = 0.45
   TERRAIN_FOREST_LOW   = 0.55
   TERRAIN_FOREST_HIGH  = 0.7
   TERRAIN_TREE         = 0.8
   TERRAIN_OREROCK      = 0.9
   NET_RENDER = False
#  SKILLS = ['exploration']
#  SKILLS = ['woodcutting', 'mining']
#  SKILLS = ['range', 'mage', 'melee']
   SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]

