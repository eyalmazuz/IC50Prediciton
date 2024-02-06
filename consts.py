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
    WANDB_KEY = "49e20ebf47e19b7061f97e1e223790d896a6b31a"


class TrainConsts:
    TRAINING_CONFIG = {
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 5e-4,
    }
    MODEL_PARAMS = {
        "emb_dim": 768,
        "dim": 768,
        "depth": 4,
        "heads": 8
    }
