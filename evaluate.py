from typing import Dict, List, Optional
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import ProteinSMILESDataset, TransformerCollate
from model import IC50Bert
from train import IC50BertTrainer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from consts import DataConsts, TrainConsts, EvalConsts, ModelParams
import wandb
from collections import defaultdict


class IC50Evaluator:
    """
    Object used to train and evaluate the IC50Bert model
    """

    def __init__(self, model: IC50Bert, dataloader: DataLoader) -> None:
        self.model = model
        self.dataloader = dataloader

    def evaluate(self) -> Dict[str, float]:
        """
        Set the model to evaluation mode and iterate through the dataloader to calculate metrics
        :return: Dict of metrics names and values
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        metrics = defaultdict(float)

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                attention_mask = batch["attention_mask"].type(torch.BoolTensor).to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(
                    ids=input_ids,
                    token_type_ids={"token_type_ids": token_type_ids},
                    mask=attention_mask,
                )

                # Update metrics
                for metric_name, metric_func in EvalConsts.METRICS.items():
                    metric_value = metric_func(
                        labels.cpu().numpy(), outputs.cpu().numpy()
                    )
                    metrics[metric_name] += metric_value

            # Calculate average metrics
            for metric_name in metrics:
                metrics[metric_name] /= len(self.dataloader)

            return metrics

    @staticmethod
    def log_metrics_to_wandb(
            metrics_dict: Dict[str, float],
            project_name: str = EvalConsts.WANDB_PROJ_NAME,
            training_config: Dict[str, float] = TrainConsts.TRAINING_CONFIG,
            run_name: str = None,
            loss_history: Optional[List] = None
    ):
        """
        This method will log the provided metrics dictionary to wandb project
        :param metrics_dict: dict with metric names as keys and float values
        :param project_name: wandb project name where metrics will be logged. default name is set in consts.py
        :param training_config: dictionary of training hyperparameter configuration to log evaluation results
        :param run_name: parameter used to name the run in wandb. default is None in which case wandb will assign a name
        :param loss_history: a List of [episodes, loss] to plot
        """
        wandb.init(project=project_name, entity='bgu-sise', config=training_config, name=run_name)
        wandb.log(metrics_dict)
        if loss_history:
            data = [[x + 1, y] for (x, y) in enumerate(loss_history)]
            loss_table = wandb.Table(data=data, columns=["episodes", "batch_loss"])
            wandb.log(
                {
                    "history_loss": wandb.plot.line(
                        loss_table, "episodes", "batch_loss", title="Average episode loss vs episode"
                    )
                }
            )
        wandb.finish()


def main() -> None:
    wandb.login(key=EvalConsts.WANDB_PROJ_NAME)
    all_metrics = []
    # Load and initialize dataloader with dataset
    data_path = os.path.join(os.getcwd(), DataConsts.DATASET_NAME)
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    collate_fn = TransformerCollate(DataConsts.TOKENIZER_FOLDER)

    # train_df, test_df = train_test_split(df.sample(frac=0.33), test_size=0.25, random_state=42)
    num_folds = EvalConsts.VALIDATION_CONFIG["num_folds"]
    num_repeats = EvalConsts.VALIDATION_CONFIG["num_repeats"]
    batch_size = TrainConsts.TRAINING_CONFIG["batch_size"]
    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(rkf.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        train_dataset = ProteinSMILESDataset(train_df)
        test_dataset = ProteinSMILESDataset(test_df)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=int(os.cpu_count()/2)
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=int(os.cpu_count()/2)
        )

        # Initialize and train model
        model = IC50Bert(
            num_tokens=collate_fn.tokenizer.vocab_size + 3,
            max_seq_len=collate_fn.tokenizer.model_max_length,
            emb_dim=ModelParams.EMBED_DIM, dim=ModelParams.DIM,
            depth=ModelParams.DEPTH, heads=ModelParams.HEADS
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=TrainConsts.TRAINING_CONFIG["learning_rate"]
        )

        trainer = IC50BertTrainer(
            model,
            train_dataloader,
            TrainConsts.TRAINING_CONFIG["num_epochs"],
            criterion,
            optimizer,
        )
        avg_episode_losses = trainer.train()

        # Initialize the evaluator and Calculate metrics
        evaluator = IC50Evaluator(model, test_dataloader)
        metrics = evaluator.evaluate()
        all_metrics.append(metrics)
        print(f"Fold {fold + 1}/{num_folds}, Repetition {fold // num_folds + 1} -\nMetrics: {metrics}")

    # Calculate mean and standard deviation of metrics across all folds and repetitions
    mean_metrics = {}
    std_metrics = {}
    for metric_name in EvalConsts.METRICS.keys():
        metric_values = [fold_metrics[metric_name] for fold_metrics in all_metrics]
        mean_metrics[metric_name] = np.mean(metric_values)
        std_metrics[metric_name] = np.std(metric_values)

    print("Mean Metrics:", mean_metrics)
    print("Std Metrics:", std_metrics)

    # Log results to wandb
    evaluator.log_metrics_to_wandb(mean_metrics, run_name="RKFold_test_run", loss_history=avg_episode_losses)

    torch.save(model.state_dict(), os.join(os.getcwd(), 'IC50Pred_Model.pt'))


if __name__ == "__main__":
    main()
