from typing import Dict, List, Optional
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from src.dataset import ProteinSMILESDataset, TransformerCollate
from src.model import IC50Bert
from src.train import IC50BertTrainer
from src.consts import DataConsts, TrainConsts, EvalConsts, ModelParams
import wandb
from collections import defaultdict
from sklearn.model_selection import train_test_split


class IC50Evaluator:
    """
    Object used to train and evaluate the IC50Bert model
    """

    def __init__(self, model: IC50Bert, dataloader: DataLoader, device: torch.device = torch.device("cuda")) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self) -> Dict[str, float]:
        """
        Set the model to evaluation mode and iterate through the dataloader to calculate metrics
        :return: Dict of metrics names and values
        """
        self.model.to(self.device)
        self.model.eval()
        metrics = defaultdict(float)

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].type(torch.BoolTensor).to(self.device)
                labels = batch["labels"].to(self.device)

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
            wandb_key: str,
            project_name: str = EvalConsts.WANDB_PROJ_NAME,
            project_entity: str = EvalConsts.WANDB_ENTITY,
            training_config: Dict[str, float] = TrainConsts.TRAINING_CONFIG,
            run_name: str = None,
    ):
        """
        This method will log the provided metrics dictionary to wandb project
        :param wandb_key: API key
        :param project_name: wandb project name where metrics will be logged. default name is set in consts.py
        :param project_entity: entity name under which to find the wandb project
        :param training_config: dictionary of training hyperparameter configuration to log evaluation results
        :param run_name: parameter used to name the run in wandb. default is None in which case wandb will assign a name
        """
        wandb.login(key=wandb_key)
        wandb.init(project=project_name, entity=project_entity, config=training_config, name=run_name)
        return wandb.run


def main() -> None:
    wandb.login(key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # Load and initialize dataloader with dataset
    data_path = os.path.join(os.getcwd(), DataConsts.DATASET_NAME)
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
    collate_fn = TransformerCollate(DataConsts.TOKENIZER_FOLDER)

    train_df, test_df = train_test_split(df, test_size=EvalConsts.VALIDATION_CONFIG["test_ratio"],
                                         random_state=42)

    train_dataset = ProteinSMILESDataset(train_df)
    test_dataset = ProteinSMILESDataset(test_df)

    batch_size = TrainConsts.TRAINING_CONFIG["batch_size"]
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=6
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=6
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
        optimizer
    )
    avg_episode_losses = trainer.train()

    # Initialize the evaluator and Calculate metrics
    evaluator = IC50Evaluator(model, test_dataloader)
    metrics = evaluator.evaluate()

    # Log results to wandb
    evaluator.log_metrics_to_wandb(metrics, run_name="train_test_split", loss_history=avg_episode_losses)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), ModelParams.MODEL_NAME))


if __name__ == "__main__":
    main()
