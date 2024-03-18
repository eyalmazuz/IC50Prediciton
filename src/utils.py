import argparse
import os
from src.consts import DataConsts, TrainConsts, EvalConsts, ModelParams


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Training args
    parser.add_argument('--train', action='store_true', help='indicate if training is required')
    parser.add_argument('--batch_size', type=int, default=TrainConsts.TRAINING_CONFIG["batch_size"],
                        help='batch size for the IC50 predictor training')
    parser.add_argument('--epochs', type=int, default=TrainConsts.TRAINING_CONFIG["num_epochs"],
                        help='number of training epochs for the IC50 predictor training')
    parser.add_argument('--learning_rate', type=float, default=TrainConsts.TRAINING_CONFIG["learning_rate"],
                        help='learning rate of IC50 predictor training')

    parser.add_argument('--train_method', choices=TrainConsts.TRAIN_METHODS,
                        default=TrainConsts.TRAIN_TEST_SPLIT, help='select method for model training')
    # train-test split parameters
    parser.add_argument('--test_ratio', type=float, default=EvalConsts.VALIDATION_CONFIG["test_ratio"],
                        help='ratio of data to take for test set when training using train_test_split')
    # cross-validation parameters
    parser.add_argument('--num_folds', type=int, default=EvalConsts.VALIDATION_CONFIG["num_folds"],
                        help='number of folds for RepeatedKFold cross-validation')
    parser.add_argument('--num_repeats', type=int, default=EvalConsts.VALIDATION_CONFIG["num_repeats"],
                        help='number of repeats for RepeatedKFold cross-validation')

    # Data args for training / evaluation
    parser.add_argument('--data_path', type=str, help='full path to dataset',
                        default=f'{os.path.join(os.getcwd(), DataConsts.IC50_DATASET_NAME)}')
    parser.add_argument('--target_metric', choices=["ic50", "kd"], default='ic50',
                        help='drug to protein interaction metric')
    parser.add_argument('--tokenizer_path', type=str, help='full path to tokenizer folder',
                        default=f'{os.path.join(os.getcwd(), DataConsts.TOKENIZER_FOLDER)}')

    default_model_path = os.path.join(os.path.join(os.getcwd(), "models"), ModelParams.MODEL_NAME)
    parser.add_argument('--save_path', type=str, default=default_model_path,
                        help='full path to save the trained model when training')
    parser.add_argument('--pretrained_path', type=str, default=default_model_path,
                        help='if a pretrained model needs to be evaluated instead of training a new one')

    # Wandb parameters to log results
    parser.add_argument('--eval', action='store_true', help='indicate if evaluation is required')
    parser.add_argument('--wandb_proj', type=str, default=EvalConsts.WANDB_PROJ_NAME,
                        help='name of wandb project to upload results')
    parser.add_argument('--wandb_key', type=str, help='wandb api key for user login')
    parser.add_argument('--wandb_entity', type=str, default=EvalConsts.WANDB_ENTITY,
                        help='wandb entity associated with the project')

    # Model parameters
    parser.add_argument('--embd_dim', type=int, default=ModelParams.EMBED_DIM,
                        help='model embedding size')
    parser.add_argument('--dim', type=int, default=ModelParams.DIM,
                        help='model dim size')
    parser.add_argument('--depth', type=int, default=ModelParams.DEPTH,
                        help='number of ltsm/decoder layers')
    parser.add_argument('--num_heads', type=int, default=ModelParams.HEADS,
                        help='number of attention heads')
    parser.add_argument('--emb_dropout', type=float, default=ModelParams.DROPOUT,
                        help='dropout rate after embedding')
    parser.add_argument('--attn_dropout', type=float, default=ModelParams.DROPOUT,
                        help='attention dropout rate')
    parser.add_argument('--layer_dropout', type=float, default=ModelParams.DROPOUT,
                        help='stochastic depth - dropout rate entire layer')
    parser.add_argument('--ff_dropout', type=float, default=ModelParams.DROPOUT,
                        help='feedforward dropout')
    parser.add_argument('--optim', choices=["adam", "adamw", 'radam'], default='radam',
                        help='Device for model training')

    # technical parameters
    parser.add_argument('--n_workers', type=int, default=min(6, os.cpu_count()),
                        help='number of workers (cpu) to use')
    parser.add_argument('--device', choices=["cuda", "cpu"], default='cuda',
                        help='Device for model training')
    return parser.parse_args()
