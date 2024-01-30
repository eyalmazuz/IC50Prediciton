from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


class DataConsts:
    DATASET_NAME = "BindingDB_EQ_IC50_Subset.tsv"
    TOKENIZER_FOLDER = "Chem_Tokenizer"


class EvalConsts:
    METRICS = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        # "MAPE": mean_absolute_percentage_error,
    }
    VALIDATION_CONFIG = {
        "num_folds": 5,
        "num_repeats": 2
    }
    WANDB_PROJ_NAME = "IC50 Prediction"


class TrainConsts:
    TRAINING_CONFIG = {
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 5e-4,
    }


class ModelParams:
    EMBED_DIM = 768
    DIM = 768
    DEPTH = 4
    HEADS = 8
