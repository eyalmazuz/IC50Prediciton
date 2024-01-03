from typing import Dict
import os
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import ProteinSMILESDataset, TransformerCollate
from model import IC50Bert
from train import IC50BertTrainer
from sklearn.model_selection import train_test_split
from consts import DataConsts, TrainConsts, EvalConsts
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
    ):
        """
        This method will log the provided metrics dictionary to wandb project
        :param metrics_dict: dict with metric names as keys and float values
        :param project_name: wandb project name where metrics will be logged. default name is set in consts.py
        :param training_config: dictionary of training hyperparameter configuration to log evaluation results
        :param run_name: parameter used to name the run in wandb. default is None in which case wandb will assign a name
        """
        wandb.init(project=project_name, config=training_config, name=run_name)
        wandb.log(metrics_dict)
        wandb.finish()


def main() -> None:
    # Load and initialize dataloader with dataset
    data_path = os.path.join(os.getcwd(), DataConsts.DATASET_NAME)
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    train_df, test_df = train_test_split(df.sample(frac=0.33), test_size=0.25, random_state=42)
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    train_dataset = ProteinSMILESDataset(train_df)
    test_dataset = ProteinSMILESDataset(test_df)

    collate_fn = TransformerCollate(DataConsts.TOKENIZER_FOLDER)
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count()
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count()
    )

    # Initialize and train model
    model = IC50Bert(
        num_tokens=len(collate_fn.tokenizer.get_vocab()),
        max_seq_len=collate_fn.tokenizer.model_max_length,
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
    trainer.train()

    # Initialize the evaluator and Calculate metrics
    evaluator = IC50Evaluator(model, test_dataloader)
    metrics = evaluator.evaluate()
    print(metrics)

    # Log results to wandb
    # evaluator.log_metrics_to_wandb(metrics, run_name="test_run")


if __name__ == "__main__":
    # TODO: implement KFold cross_validation
    main()
