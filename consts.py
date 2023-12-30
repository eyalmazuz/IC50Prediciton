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
        "MAPE": mean_absolute_percentage_error,
    }

    WANDB_PROJ_NAME = "IC50 Prediction"


class TrainConsts:
    TRAINING_CONFIG = {
        "num_epochs": 10,
        "learning_rate": 0.001,
    }
