import os
import sys
from typing import Dict
from datetime import datetime
from time import sleep
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from src.dataset import ProteinSMILESDataset, TransformerCollate, ProteinSMILESDavisDataset
from src.model import IC50Bert
from src.train import IC50BertTrainer
from src.evaluate import IC50Evaluator
from src.utils import parse_arguments
from src.consts import TrainConsts, EvalConsts

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold


def get_dataloader(data: pd.DataFrame, collate_func: TransformerCollate, args, test: bool = False):
    index_reset_data = data.reset_index()
    ic50_metric = True if args.target_metric == "ic50" else False
    if args.data_path.endswith(".csv"):
        dataset = ProteinSMILESDavisDataset(index_reset_data)
    else:
        dataset = ProteinSMILESDataset(index_reset_data, ic50_metric=ic50_metric, kd_metric=not ic50_metric)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if test else True,
        collate_fn=collate_func,
        # num_workers=args.n_workers,
        pin_memory=True
    )
    return dataloader


def get_model_and_optimizer(collate_func: TransformerCollate, args) -> (IC50Bert, torch.optim):
    # Initialize model
    model = IC50Bert(
        num_tokens=collate_func.tokenizer.vocab_size + 3,
        max_seq_len=collate_func.tokenizer.model_max_length,
        emb_dim=args.embd_dim, dim=args.dim, depth=args.depth, heads=args.num_heads,
        emb_dropout=args.emb_dropout, attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout, layer_dropout=args.layer_dropout
    )

    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-05
        )

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-05
        )

    else:
        optimizer = torch.optim.RAdam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-05
        )

    return model, optimizer


def train_ic50_predictor(
        train_df: pd.DataFrame, collate_func: TransformerCollate, args, val_df: pd.DataFrame | None = None, wandb_run=None
) -> (IC50Bert, Dict):
    criterion = torch.nn.MSELoss()
    train_dataloader = get_dataloader(train_df, collate_func, args, test=False)
    val_dataloader = None if val_df is None else get_dataloader(val_df, collate_func, args, test=True)
    model, optimizer = get_model_and_optimizer(collate_func, args)

    trainer = IC50BertTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(args.device),
        wandb_run=wandb_run
    )
    avg_episode_losses = trainer.train()
    return model, avg_episode_losses


def evaluate_ic50_predictor(model: IC50Bert, test_df: pd.DataFrame, collate_func: TransformerCollate, args) -> Dict:
    test_dataloader = get_dataloader(test_df, collate_func, args, test=True)
    evaluator = IC50Evaluator(model, test_dataloader, device=torch.device(args.device))
    metrics = evaluator.evaluate()
    return metrics


def main():
    args = parse_arguments()

    wandb_run = None
    if args.eval:
        train_method = args.train_method if args.train else "pretrained"
        train_conf = {
            "dataset_name": os.path.basename(args.data_path),
            "target_metric": args.target_metric,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_method": train_method
        }

        wandb_run = IC50Evaluator.log_metrics_to_wandb(
            wandb_key=args.wandb_key,
            project_name=args.wandb_proj,
            project_entity=args.wandb_entity,
            training_config=train_conf,
            run_name=f'{args.target_metric + "_metric_" + str(datetime.now().strftime("%m_%d_%H_%M_%S"))}'
        )

    if args.data_path.endswith(".tsv"):
        ic50_data = pd.read_csv(args.data_path, sep="\t", low_memory=False)
    elif args.data_path.endswith(".csv"):
        ic50_data = pd.read_csv(args.data_path, low_memory=False)
    else:
        raise ValueError("Data isn't a valid format")
    collate_func = TransformerCollate(args.tokenizer_path)
    metrics_dict = {}
    avg_episode_losses = {}

    if args.device == "cuda":
        if torch.cuda.is_available():
            print("cuda detected, training on GPU\n")
        else:
            print("cuda device specified but wasn't detected - defaulting to cpu\n")
            args.device = "cpu"

    if args.train:
        if args.train_method == TrainConsts.TRAIN_TEST_SPLIT:
            train_val, test = train_test_split(ic50_data, test_size=args.test_ratio, random_state=42, shuffle=True)
            train, val = train_test_split(train_val, test_size=args.test_ratio, random_state=42, shuffle=True)
            model, avg_episode_losses = train_ic50_predictor(train, collate_func, args, val, wandb_run)
            metrics_dict = evaluate_ic50_predictor(model, test, collate_func, args)

        if args.train_method == TrainConsts.CROSS_VAL:
            all_metrics = []
            rkf = RepeatedKFold(n_splits=args.num_folds, n_repeats=args.num_repeats, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(rkf.split(ic50_data)):
                repetition = fold // args.num_folds
                train, test = ic50_data.iloc[train_idx], ic50_data.iloc[test_idx]

                model, avg_episode_losses = train_ic50_predictor(train, collate_func, args, wandb_run)
                metrics = evaluate_ic50_predictor(model, test, collate_func, args)
                print(f"\nFold {fold + 1 - args.num_folds * repetition }/{args.num_folds}, "
                      f"Repetition {repetition + 1} -\nMetrics: {metrics}")

                all_metrics.append(metrics)

            # Calculate mean and standard deviation of metrics across all folds and repetitions
            for metric_name in EvalConsts.METRICS.keys():
                metric_values = [fold_metrics[metric_name] for fold_metrics in all_metrics]
                metrics_dict[metric_name] = np.mean(metric_values)

        # save generated model
        save_folder = os.path.dirname(args.save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

        # save generation command
        with open(f'{save_folder}/howto.txt', 'w') as f:
            f.write(' '.join(sys.argv))

    # if train is False and model pretrained_path is provided to evaluate
    elif args.pretrained_path:
        model, _ = get_model_and_optimizer(collate_func, args)
        model.load_state_dict(torch.load(args.pretrained_path))
        metrics_dict = evaluate_ic50_predictor(model, ic50_data, collate_func, args)

    # Log results to wandb
    if args.eval and metrics_dict:
        wandb.log(metrics_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run('main()', sort='cumtime')
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    #     main()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# TODO: 1. Add option to read model and training configurations from json instead of editing consts.py
# TODO: 2. raise exceptions for relevant args input errors
