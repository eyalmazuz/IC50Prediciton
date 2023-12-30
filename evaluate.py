from typing import Dict
from numpy.typing import ArrayLike
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import ProteinSMILESDataset, TransformerCollate
from model import IC50Bert
from consts import *
import wandb


class IC50Evaluator:
    """
    Object used to train and evaluate the IC50Bert model
    """

    def __init__(self, model: IC50Bert, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader

    def evaluate(self) -> (ArrayLike, ArrayLike):
        """
        Set the model to evaluation mode and iterate through the dataloader to get model predictions
        :return: Tuple of ArrayLikes - (True labels, Model Predictions)
        """
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                outputs = self.model(*inputs)

                all_labels.extend(labels.numpy())
                all_predictions.extend(outputs.numpy())

        return all_labels, all_predictions

    def calculate_metrics(self) -> Dict[str, float]:
        labels, predictions = self.evaluate()

        eval_metrics = {}
        for metric_name, metric_func in EvalConsts.METRICS.items():
            metric_value = metric_func(labels, predictions)
            eval_metrics[metric_name] = metric_value
            print(f"{metric_name}: {metric_value:.4f}")

        return eval_metrics

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
    data_path = os.path.join(os.getcwd(), DataConsts.dataset_name)
    df = pd.read_csv(data_path, sep="\t", low_memory=False)

    dataset = ProteinSMILESDataset(df)
    collate_fn = TransformerCollate("Chem_Tokenizer")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize and train model
    model = IC50Bert()

    # TODO: Need to train the model
    # model.train(model, dataloader, num_epochs=10, learning_rate=0.001)

    # Initialize the evaluator
    evaluator = IC50Evaluator(model, dataloader)

    # Calculate metrics
    metrics = evaluator.calculate_metrics()

    # Log results to wandb
    # evaluator.log_metrics_to_wandb(metrics, run_name="test_run")


if __name__ == "__main__":
    # TODO: implement KFold cross_validation
    main()
