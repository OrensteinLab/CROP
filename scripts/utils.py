DEFAULT_BATCH_SIZE = 512
DEFAULT_PATHWAYS_CHANNELS = 8


N_EPOCHS_FIRST_PHASE = 50 # was 50 
N_EPOCHS_SECOND_PHASE = 10 # was 10

ALPHA_START = 0.00
ALPHA_DECAY_EPOCHS = 20

LEFT_CONTEXT = 50 # was 100
RIGHT_CONTEXT = 50 # was 100
PAM_LEN = 3
MAX_SEQ_LEN = LEFT_CONTEXT + PAM_LEN + RIGHT_CONTEXT  # 203

MIN_DELTA = -99 # was -100
MAX_DELTA = 88 # was 100
NUM_CLASSES = MAX_DELTA - MIN_DELTA + 1  # 201
DELTA_OFFSET = -MIN_DELTA  # so -100->0

# Vocabulary for tokens
VOCAB = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
VOCAB_SIZE = len(VOCAB)

G_MAX = 8 # max left/right gap for MH calculation


# Dara splits
TRAIN_TEST_RATIO = 0.9
N_TRAIN_VAL_FOLDS = 9







########################################
# DATASETS CONFIGURATION
########################################

DATASET_CONFIGS = [
    {
        "csv_path": "data/FORECasT_K562.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/FORECasT_mESC.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },

        {
        "csv_path": "data/FORECasT_BOB.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
        {
        "csv_path": "data/FORECasT_CHO.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/FORECasT_HAP1.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/FORECasT_RPE1.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/FORECasT_K562_TREX2.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/FORECasT_K562_2A_TREX2.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
            {
        "csv_path": "data/FORECasT_K562_eCAS9.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },



    {
        "csv_path": "data/SPROUT_T.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },   



    {
        "csv_path": "data/ALDIT_K562.csv",
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/ALDIT_Jurkat.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/ALDIT_HAP1.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },  
    {
        "csv_path": "data/ALDIT_K562_DNTTOE.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },   
    {
        "csv_path": "data/ALDIT_Jurkat_DNTTKO.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },   




    {
        "csv_path": "data/XCRISP_inDelphi_mESC.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },
    {
        "csv_path": "data/XCRISP_inDelphi_mESC_NHEJdeficient.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    }, 
    {
        "csv_path": "data/XCRISP_inDelphi_U2OS.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },



    {
        "csv_path": "data/XCRISP_FORECasT_mESC.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    }, 
    {
        "csv_path": "data/XCRISP_FORECasT_HAP1.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    }, 
    {
        "csv_path": "data/XCRISP_FORECasT_TREX2.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },                


    {
        "csv_path": "data/Lindel_HEK293T.csv", 
        "sequence_col": "sequence",
        "pam_index_col": "PAM position",
    },  


]

index_to_DS_name = {0: "FORECasT K562",
                    1: "FORECasT mESC",
                    2: "FORECasT BOB",
                    3: "FORECasT CHO",
                    4: "FORECasT HAP1",
                    5: "FORECasT RPE1",
                    6: "FORECasT TREX2",
                    7: "FORECasT 2A TREX2",
                    8: "FORECasT eCAS9",
                    9: "SPROUT T",
                    10: "Aldit K562",
                    11: "Aldit Jurkat",
                    12: "Aldit HAP1",
                    13: "Aldit K562 DNTTOE",
                    14: "Aldit Jurkat DNTTKO",
                    15: "InDelphi mESC",
                    16: "InDelphi mESC NHEJdeficient",
                    17: "InDelphi U2OS",
                    18: "XCRISP FORECasT mESC",
                    19: "XCRISP FORECasT HAP1",
                    20: "XCRISP FORECasT TREX2",
                    21: "Lindel HEK293T",}
index_to_percentage_removed = {
                    0: 0.0, # was 20.0
                    1: 0.0, # was 20.0
                    2: 0.0, # was 20.0
                    3: 0.0, # was 20.0
                    4: 0.0, # was 20.0
                    5: 0.0, # was 30.0
                    6: 0.0, 
                    7: 0.0,
                    8: 0.0,
                    9: 0.0,
                    10: 0.0,
                    11: 0.0,
                    12: 0.0,
                    13: 0.0,
                    14: 0.0,
                    15: 0.0,
                    16: 0.0,
                    17: 0.0,
                    18: 0.0,
                    19: 0.0,
                    20: 0.0,
                    21: 0.0,
}


SELECTED_IDS = {0,1,2,3, 4,5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15 ,16, 17}   # All datasets except XCRISP FORECAST and Lindel

GROUPING_RULES = [
    {0,1,2,3,4,5,6,7,8},   # FORECasT
    # sprout (9) will automatically become a single-dataset group
    {10,11,12, 13, 14},         # Aldit
    {15,16,17},      # InDelphi
    # Lindel (21) will automatically become a single-dataset group
]







## COMPACT EVERYTHING
old_selected_ids = set(SELECTED_IDS)

N_DATASETS_PICKED = len(old_selected_ids)

# Filter dataset configs
DATASET_CONFIGS = [
    cfg for i, cfg in enumerate(DATASET_CONFIGS)
    if i in old_selected_ids
]

# Remap names
index_to_DS_name = {
    new_i: index_to_DS_name[old_i]
    for new_i, old_i in enumerate(sorted(old_selected_ids))
}

# Remap removal percentages
index_to_percentage_removed = {
    new_i: index_to_percentage_removed[old_i]
    for new_i, old_i in enumerate(sorted(old_selected_ids))
}

# Old → new index mapping
old_to_new_index = {
    old_i: new_i
    for new_i, old_i in enumerate(sorted(old_selected_ids))
}

# Replace SELECTED_IDS with compacted IDs
SELECTED_IDS = set(range(N_DATASETS_PICKED))

# Remap grouping rules
GROUPING_RULES = [
    {old_to_new_index[old_i] for old_i in group if old_i in old_selected_ids}
    for group in GROUPING_RULES
    if any(old_i in old_selected_ids for old_i in group)
]



