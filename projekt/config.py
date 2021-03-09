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

   v                       = False

   ENV_NAME                = 'Neural_MMO'
   ENV_VERSION             = '1.5'
   NUM_WORKERS             = 6
   NUM_GPUS_PER_WORKER     = 0
   NUM_GPUS                = 1
   TRAIN_BATCH_SIZE        = 4800 # to match evo, normally 4000
   #TRAIN_BATCH_SIZE        = 400
   ROLLOUT_FRAGMENT_LENGTH = 100
   SGD_MINIBATCH_SIZE      = 128
   NUM_SGD_ITER            = 1

   MODEL        = 'current'
   SCRIPTED_BFS = False
   SCRIPTED_DP  = False
   EVALUATE     = False
   LOCAL_MODE   = False

   # Model dimensions
   EMBED  = 64
   HIDDEN = 64

   # Environment parameters
   NPOP = 1    # Number of populations #SET SHARE POLICY TRUE
   NENT = 1024 # Maximum population size
   NMOB = 1024 # Number of NPCS

   NMAPS = 256 # Number maps to generate

   #Horizons for training and evaluation
   #TRAIN_HORIZON      = 500 #This in in agent trajs
   TRAIN_HORIZON      = 1000 #This in in agent trajs
   EVALUATION_HORIZON = 2048 #This is in timesteps

   #Agent vision range
   STIM    = 7

   #Maximum number of observed agents
   N_AGENT_OBS = 100

   # Whether to share weights across policies
   # The 1.4 baselines use one policy
   POPULATIONS_SHARE_POLICIES = False
   NPOLICIES = 1 if POPULATIONS_SHARE_POLICIES else NPOP

   #Overlays
   OVERLAY_GLOBALS = False

   #Evaluation
   LOG_DIR = 'experiment/'
   LOG_FILE = 'evaluation.npy'
   LOG_FIGURE = 'evaluation.html'

   #Visualization
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
   NMOB                    = 0

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


ALL_SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
COMBAT_SKILLS = ['range', 'mage', 'melee']
EXPLORE_SKILLS = ['exploration']
HARVEST_SKILLS = ['woodcutting', 'mining']

class TreeOrerock(SmallMap):
   NEW_EVAL = False
   EVO_MAP = True
   FIXED_MAPS = True
   EVALUATE = True
   NENT                 = 16
   NMOB                 = 0
   MODEL                = 'current'
   TERRAIN_SIZE         = 70
   TERRAIN_DIR          = Config.TERRAIN_DIR_SMALL
   ROOT                 = os.path.join(os.getcwd(), TERRAIN_DIR, 'map')
#  TERRAIN_RENDER       = True
#  TERRAIN_ALPHA = 0
#  TERRAIN_BETA = 0
#  TERRAIN_WATER        = 0.25
#  TERRAIN_FOREST_LOW   = 0.35
   TERRAIN_GRASS_0   = 0.4
   TERRAIN_LAVA  = 0.45
   TERRAIN_SPAWN = 0.5
#  TERRAIN_GRASS        = 0.7
#  TERRAIN_FOREST_HIGH  = 0.725
   TERRAIN_TREE         = 0.8
   TERRAIN_OREROCK      = 0.85
   GRIDDLY = False
   SKILLS               = ALL_SKILLS
   FITNESS_METRIC       = 'L2'
   MAP = 'PCG'
   INFER_IDX = 0
   N_EVAL = 20
   EVO_VERBOSE          = True
   EVO_SAVE_INTERVAL    = 5
   GRIDDLY = False


ALL_SKILLS = ['constitution', 'fishing', 'hunting', 'range', 'mage', 'melee', 'defense', 'woodcutting', 'mining', 'exploration',]
COMBAT_SKILLS = ['range', 'mage', 'melee']
EXPLORE_SKILLS = ['exploration']
HARVEST_SKILLS = ['woodcutting', 'mining']

class EvoNMMO(TreeOrerock):

   FIXED_MAPS = False
   EVALUATE = False
 # INFER_IDX = 79766
 # INFER_IDX = 80117
   # How to measure diversity of agents on generated map.
   FITNESS_METRIC = 'L2' # 'Differential', 'L2', 'Discrete', 'Hull', 'Sum', 'Lifespans', 'Actions'
   GENOME = 'Random'  # CPPN, Pattern, Random
   THRESHOLD = False
   TERRAIN_MODE = 'contract'
   EVO_MAP = True
   RENDER = False
   MODEL = 'current'
   NENT = 16  # Maximum population size
   TERRAIN_SIZE = 70
   EVO_DIR = 'all_random_l2_1'
#  ROOT = os.path.join(os.getcwd(), 'evo_experiment', EVO_DIR, 'maps', 'map')
   N_EVO_MAPS = 48
   MAX_STEPS = 100
   MATURE_AGE = 3
   ROLLING_FITNESS = 25  # Size of window to use while calculating mean rolling fitness
   TERRAIN_RENDER = False
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
   SKILLS = ALL_SKILLS
   EVO_ALGO = 'Simple'  # Simple, MAP-Elites, NEAT
   N_PROC = 6
   PRETRAINED = False
#  MAP_DIMS = ['woodcutting', 'mining']
   ME_DIMS = ['mining', 'woodcutting']
   ME_BIN_SIZES = [20, 20]
   ME_BOUNDS = [(0, 100), (0, 100)]
   ARCHIVE_UPDATE_WINDOW = 15  # How long without any updates to ME archive before randomly re-evaluating some elites?
   FEATURE_CALC = 'map entropy'
   TEST = False
   ITEMS_PER_BIN = 1

class Explore(EvoNMMO):
   SKILLS = EXPLORE_SKILLS

class Combat(EvoNMMO):
   SKILLS = COMBAT_SKILLS

class All(EvoNMMO):
   SKILLS = ALL_SKILLS

class Griddly(EvoNMMO):
   TRAIN_RENDER = False
   GRIDDLY = True
   REGISTERED = False  #FIXME: hack. Do not set this.
   TEST = False
   NENT=5
   FROZEN = False
   PRETRAIN = False
   EVO_DIR = 'griddly_scratch_0'
   TERRAIN_BORDER = 1
   ME_DIMS = ['mine_skill', 'woodcut_skill']
   ME_BIN_SIZES = [100, 100]
   ME_BOUNDS = [(0, 50), (0, 50)]
   SKILLS = ['drink_skill', 'gather_skill', 'woodcut_skill', 'mine_skill']
