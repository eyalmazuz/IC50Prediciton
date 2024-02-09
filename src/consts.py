from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


class DataConsts:
    DATASET_NAME = "Filtered_BindingDB_EQ_IC50_Subset.tsv"
    TOKENIZER_FOLDER = "Chem_Tokenizer"


class EvalConsts:
    METRICS = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        # "MAPE": mean_absolute_percentage_error,
    }
    VALIDATION_CONFIG = {
        "num_folds": 5,
        "num_repeats": 2,
        "test_ratio": 0.2
    }

    WANDB_PROJ_NAME = "IC50 Prediction"
    WANDB_KEY = "49e20ebf47e19b7061f97e1e223790d896a6b31a"
    WANDB_ENTITY = "bgu-sise"


class TrainConsts:
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VAL = "cross_validation"

    TRAIN_METHODS = {
        TRAIN_TEST_SPLIT,
        CROSS_VAL
    }

    TRAINING_CONFIG = {
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 0.001,
    }


class ModelParams:
    EMBED_DIM = 768
    DIM = 768
    DEPTH = 4
    HEADS = 8
    DROPOUT = 0.1

    MODEL_NAME = "IC50Pred_Model.pt"
