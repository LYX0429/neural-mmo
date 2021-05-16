# NO SIM, EVO ALGO and MAP REPRESENTATION tests

## ES

mapText_tileFlip_ES:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_tileFlip_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 1000\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

mapText_prims_ES:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_prims_ES_3\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

mapText_CPPN_ES:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_CPPN_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

mapText_CA_ES:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_CA_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME CA\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

mapText_CA_CMAES:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_CA_CMAES_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO CMAES\
		--GENOME CA\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

## ME

mapText_tileFlip_ME:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_tileFlip_ME_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ME_BIN_SIZES=[10,10]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 1000\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC='map entropy'

mapText_prims_ME:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_prims_ME_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 2\
		--ME_BIN_SIZES=[10,10]\
		--ME_BOUNDS="[(0,100), (0, 100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC='map entropy'

mapText_CPPN_ME:
	python ForgeEvo.py\
		--TEST False\
		--EVO_DIR MapTestText_CPPN_ME_0\
		--N_PROC 12\
		--N_EVO_MAPS 100\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC MapTestText\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN 1
		--ME_BIN_SIZES=[10,10]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC='map entropy'

## GRIDDLY

### Pre-train on random maps

gdy_pretrain:
	python nmmo_evo.py\
	    --EVO_DIR gdy_pretrain_7\
	    --TERRAIN_SIZE 18\
	    --NENT 3\
	    --N_PROC 6\
	    --N_EVO_MAPS 12\
	    --GENOME Random\
	    --EVO_ALGO 'MAP-Elites'\
	    --FITNESS_METRIC Scores\
	    --FEATURE_CALC 'map entropy'\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
        --PRETRAIN True\
        --EVO_SAVE_INTERVAL 100\
        --TRAIN_RENDER False\
        --ARCHIVE_UPDATE_WINDOW 0\
        --MODEL None


render_gdy_pretrain:
	python nmmo_evo.py\
		--EVO_DIR gdy_pretrain_7 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--FITNESS_METRIC Scores\
		--RENDER True\
		--NENT 3

### Train frozen

gdy_frz_score_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_score_tileFlip_ES_1\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC Scores\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=10\
		--ROLLING_FITNESS=10\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False\
		--FROZEN True\
		--PRETRAINED True\
		--MODEL 'gdy_pretrain_7'

render_gdy_frz_score_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_score_tileFlip_ES_0 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--TRAIN_HORIZON 200\
		--MAX_STEPS 200\
		--NENT 3

gdy_frz_score_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_score_prims_ES_0\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC Scores\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--ROLLING_FITNESS=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False\
		--FROZEN True\
		--PRETRAINED True\
		--MODEL 'gdy_pretrain_7'

render_gdy_frz_score_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_score_prims_ES_0 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--NENT 3

gdy_frz_up_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_up_prims_ES_0\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC y_deltas\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--ROLLING_FITNESS=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False\
		--FROZEN True\
		--PRETRAINED True\
		--MODEL 'gdy_pretrain_7'

render_gdy_frz_up_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_frz_up_prims_ES_0 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--NENT 3


### Skills

gdy_skill_CPPN_ES:
	python nmmo_evo.py\
	python nmmo_evo.py\
		--GRIDDLY True\
		--TEST False\
		--EVO_DIR gdy_skill_CPPN_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 1\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Sum\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 30\
		--NENT 5\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW 0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_skill_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_skill_CPPN_ES_1 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 30\
		--NENT 5

### Score (flexible, for debugging for now)

gdy_score_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_score_CPPN_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Scores\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 17\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_score_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_score_CPPN_ES_1 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 17\
		--NENT 3

gdy_up_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_CPPN_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC y_deltas\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 17\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_up_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_CPPN_ES_0 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 17\
		--NENT 3

gdy_up_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_prims_ES_3\
		--N_PROC 12\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC y_deltas\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 17\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--ROLLING_FITNESS=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_up_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_prims_ES_2 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 17\
		--NENT 3

gdy_up_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_tileFlip_ES_2\
		--N_PROC 12\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC y_deltas\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--ROLLING_FITNESS=1\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_up_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_up_tileFlip_ES_2 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--NENT 3


gdy_div_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_div_prims_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC Differential\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--ROLLING_FITNESS=1\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

### Lifespans

gdy_life_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_tileFlip_ME_1\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC Lifespans\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 30\
		--NENT 1\
		--FEATURE_CALC=None\
		--ARCHIVE_UPDATE_WINDOW00\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER=False

render_gdy_life_tileFlip_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_tileFlip_ME_1 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 30\
		--NENT 1 


gdy_life_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_CPPN_ES_7\
		--N_PROC 0\
		--N_EVO_MAPS 24\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Lifespans\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 17\
		--NENT 3\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--ARCHIVE_UPDATE_WINDOW=0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER Render

render_gdy_life_CPPN_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_CPPN_ES_5 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 17\
		--NENT 2  

gdy_life_CPPN_ME:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_CPPN_ME_8\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Lifespans\
		--ITEMS_PER_BIN=2\
		--ME_BIN_SIZES=[10,10]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 30\
		--NENT 8\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_life_CPPN_ME:
	python nmmo_evo.py\
		--EVO_DIR gdy_life_CPPN_ME_8 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 30\
		--NENT 8

gdy_ALP_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_ALP_prims_ES_2\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC ALP\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--FEATURE_CALC None\
		--ARCHIVE_UPDATE_WINDOW 0\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

render_gdy_ALP_prims_ES:
	python nmmo_evo.py\
		--EVO_DIR gdy_ALP_prims_ES_2 \
		--RENDER True\
		--TRAIN_RENDER True\
		--TERRAIN_SIZE 18\
		--NENT 3

gdy_ALP_CPPN_ME:
	python nmmo_evo.py\
		--EVO_DIR gdy_ALP_CPPN_ME_0\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC ALP\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 10\
		--TERRAIN_SIZE 30\
		--NENT 8\
		--FEATURE_CALC='map entropy'\
		--SKILLS=['drink_skill','gather_skill','woodcut_skill','mine_skill']\
		--TERRAIN_RENDER False\
		--TRAIN_RENDER False

life_prims_ME:
	python ForgeEvo.py \
	--EVO_DIR life_prims_ME_0 \
	--FITNESS_METRIC Lifespans \
	--GENOME Pattern \
	--EVO_ALGO MAP-Elites \
	--FEATURE_CALC "map entropy" \
	--ME_BIN_SIZES=[100] \
	--ME_BOUNDS="[(25,100)]" \
	--TERRAIN_SIZE 70 \
	--N_PROC 12 \
	--N_EVO_MAPS 12 \
	--NENT 3 \
	--TERRAIN_RENDER False \
	--EVO_SAVE_INTERVAL 50 \
	--ROLLING_FITNESS 10


# PROSPEROUS MAPS

life_tileFlip_ES:
	python ForgeEvo.py\
		--EVO_DIR life_tileFlip_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC Lifespans\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]"\
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--NENT 3\
		--FEATURE_CALC=None

life_CPPN_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR life_CPPN_ES_3\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Lifespans\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 50\
		--FEATURE_CALC=None

render_life_CPPN_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --MAP life_CPPN_ES_3\
		--MODEL life_CPPN_ES_3\
		--NENT 3\
		--INFER_IDX "(0,0,0)"


div_CPPN_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_CPPN_ES_0\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

render_div_CPPN_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_CPPN_ES_0\
        --MAP div_CPPN_ES_0\
		--MODEL div_CPPN_ES_0\
		--NENT 3\
		--TERRAIN_SIZE 36\
		--INFER_IDX "(0,0,0)"


div_xplor_CPPN_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_xplor_CPPN_ES_0\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC L2\
		--SKILLS="['exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

render_div_xplor_CPPN_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_xplor_CPPN_ES_0\
        --MAP div_xplor_CPPN_ES_0\
		--MODEL div_xplor_CPPN_ES_0\
		--NENT 3\
		--TERRAIN_SIZE 36\
		--INFER_IDX "(0,0,0)"


div_xplor_prims_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_xplor_prims_ES_0\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

render_div_xplor_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_xplor_prims_ES_0\
        --MAP div_xplor_prims_ES_0\
		--MODEL div_xplor_prims_ES_0\
		--NENT 3\
		--TERRAIN_SIZE 36\
		--INFER_IDX "(0,0,0)"

div_pair_harvest_prims_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_pair_harvest_prim_ES_1\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['woodcutting','mining']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

render_div_pair_harvest_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_pair_harvest_prim_ES_1\
        --MAP div_pair_harvest_prim_ES_1\
		--MODEL div_pair_harvest_prim_ES_1\
		--NENT 3\
		--TERRAIN_SIZE 36\
		--INFER_IDX "(0,0,0)"

div_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_pair_prim_ES_2\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_div_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_pair_prim_ES_2\
        --MAP div_pair_prim_ES_2\
		--MODEL div_pair_prim_ES_2\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"


### MARCH 24, 2021 ############################################################

div_xplor_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_xplor_pair_prims_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_xplor_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_xplor_pair_prims_ES_0\
        --MAP div_xplor_pair_prims_ES_0\
		--MODEL div_xplor_pair_prims_ES_0\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

div_harvest_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_harvest_pair_prims_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['mining', 'woodcutting']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_harvest_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_harvest_pair_prims_ES_0\
        --MAP div_harvest_pair_prims_ES_0\
		--MODEL div_harvest_pair_prims_ES_0\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"


div_all_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_all_pair_prims_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_all_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_all_pair_prims_ES_1\
        --MAP div_all_pair_prims_ES_1\
		--MODEL div_all_pair_prims_ES_1\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

div_all_pair_cppn_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_all_pair_cppn_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_all_pair_cppn_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_all_pair_cppn_ES_1\
        --MAP div_all_pair_cppn_ES_1\
		--MODEL div_all_pair_cppn_ES_1\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

div_all_pair_tile_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR div_all_pair_tile_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Random\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None

render_all_pair_tile_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_all_pair_tile_ES_1\
        --MAP div_all_pair_tile_ES_1\
		--MODEL div_all_pair_tile_ES_1\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"


### SINGLE-SPAWN

onespawn_div_xplor_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR onespawn_div_xplor_pair_prims_ES_0\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None\
		--SINGLE_SPAWN=True

render_onespawn_xplor_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR onespawn_div_xplor_pair_prims_ES_0\
        --MAP onespawn_div_xplor_pair_prims_ES_0\
		--MODEL onespawn_div_xplor_pair_prims_ES_0\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

# start from one spawn, maximize diversity in combat skills
onespawn_div_combat_pair_prims_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR onespawn_div_combat_pair_prims_ES_0\
		--N_PROC 6\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC L2\
		--SKILLS="['range','mage','melee']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None\
		--SINGLE_SPAWN=True

render_onespawn_div_combat_pair_prims_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR onespawn_div_combat_pair_prims_ES_0\
        --MAP onespawn_div_combat_pair_prims_ES_0\
		--MODEL onespawn_div_combat_pair_prims_ES_0\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

# PAIRED type algorithm because why not
paired_ES:
	python ForgeEvo.py\
		--NENT 8\
		--EVO_DIR paired_ES_1\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME Pattern\
		--FITNESS_METRIC Lifespans\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 70\
		--FEATURE_CALC=None\
		--NPOLICIES=2\
		--PAIRED=True

render_paired_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR paired_ES_1\
        --MAP paired_ES_1\
		--MODEL paired_ES_1\
		--NPOLICIES=2\
		--PAIRED=True\
		--NENT 8\
		--TERRAIN_SIZE 70\
		--INFER_IDX "(0,0,0)"

### PRETRAIN

pretrain_vanilla:
	python Forge.py train\
		--config TreeOrerock\
		--MODEL None\
		--TRAIN_HORIZON 100\
		--NUM_WORKERS 12

pretrain_diversity:
	python Forge.py train\
		--config TreeOrerock\
		--MODEL None\
		--REWARD_DIVERSITY=True\
		--TRAIN_HORIZON 100\
		--NUM_WORKERS 12

################################################################################



div_pair_CPPN_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_pair_CPPN_ES_1\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC L2\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"\
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

render_div_pair_CPPN_ES:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --EVO_DIR div_pair_CPPN_ES_0\
        --MAP div_pair_CPPN_ES_0\
		--MODEL div_pair_CPPN_ES_0\
		--NENT 3\
		--TERRAIN_SIZE 36\
		--INFER_IDX "(0,0,0)"

div_CPPN_ME:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR div_CPPN_ME_3\
		--N_PROC 0\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Differential\
		--SKILLS="['constitution','fishing','hunting','range','mage','melee','defense','woodcutting','mining','exploration']"
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 50\
		--FEATURE_CALC=None

render_div_CPPN_ME:
	cd ../neural-mmo-client &&\
	./UnityClient/neural-mmo-resources.x86_64 &\
    python Forge.py render --config TreeOrerock\
        --MAP div_CPPN_ME_3\
		--MODEL div_CPPN_ME_3\
		--EVO_DIR div_CPPN_ME_3\
		--TERRAIN_SIZE 18\
		--NENT 3\
		--INFER_IDX "(0,0,0)"

sum_xplor_CPPN_ES:
	python ForgeEvo.py\
		--NENT 3\
		--EVO_DIR sum_xplor_CPPN_ES_0\
		--N_PROC 12\
		--N_EVO_MAPS 12\
		--EVO_ALGO MAP-Elites\
		--GENOME CPPN\
		--FITNESS_METRIC Sum\
		--SKILLS=['exploration']
		--TERRAIN_RENDER False\
		--ITEMS_PER_BIN=12\
		--ME_BIN_SIZES=[1,1]\
		--ME_BOUNDS="[(0,100),(0,100)]" \
		--EVO_SAVE_INTERVAL 100\
		--TERRAIN_SIZE 36\
		--FEATURE_CALC=None

eval_pattern:
	python Forge.py evaluate --config TreeOrerock \
	--EVALUATION_HORIZON 1000 \
	--N_EVAL 20 --NEW_EVAL \
   --MAP greene/all_pattern_l2_5 \
  --INFER_IDX 37 \
 --MODEL evo_experiment/greene/all_pattern_l2_5
# -- MODEL evo_experiment/greene/all_random_l2_2
# --MODEL evo_experiment/greene/all_random_l2_2
# --MODEL all_ME_harvest_l2_0 

eval_cppn:
	python Forge.py evaluate --config TreeOrerock \
	--EVALUATION_HORIZON 1000 \
	--N_EVAL 20 --NEW_EVAL \
   --MAP greene/all_cppn_l2_11 \
	--INFER_IDX 157\
   --MODEL evo_experiment/greene/all_pattern_l2_5
#--MODEL current

eval_tile:
	python Forge.py evaluate --config TreeOrerock \
	--EVALUATION_HORIZON 1000 \
	--N_EVAL 20 --NEW_EVAL \
   --MAP greene/all_random_diff_2 \
	--INFER_IDX 47\
   --MODEL evo_experiment/greene/all_random_diff_2
#--MODEL current

compare:
	python plot_diversity.py \
		--compare_models all_random_l2_2 all_random_disc_1 all_pattern_l2_5 all_ME_harvest_l2_0 all_cppn_l2_11 current \
		--infer_idx 37 \
		--map all_random_l2_2
