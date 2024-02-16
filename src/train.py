from src.model import IC50Bert
import torch
from typing import List, Dict, Tuple
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopper:
    """Early stopping functionality for training"""
    def __init__(self, patience: int = 1, min_delta: float = 0.0, min_stopping_episode: int = 0):
        """
        :param patience: Number of episodes without improvement to wait before stopping
        :param min_delta: tolerance around min validation, if within tolerance counter doesn't increase
        :param min_stopping_episode: early stopping isn't considered before this episode
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_stopping_episode = min_stopping_episode
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float, episode: int) -> bool:
        """
        Consider early stopping
        :param validation_loss: validation loss of the current episode
        :param episode: number of current episode
        :return: Boolean signal indicating if early stopping criteria have been met
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and episode >= self.min_stopping_episode:
                return True

        return False


class IC50BertTrainer:
    """
    Class used in the training of an IC50Bert model
    """

    def __init__(
        self,
        model: IC50Bert,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        num_epochs: int,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        wandb_run=None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model = model
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.wandb_run = wandb_run
        self.device = device
        self.early_stopper = EarlyStopper(patience=10, min_delta=0.1, min_stopping_episode=10)

    def train(self) -> Dict[str, List[float]]:
        """
        Train the specified model using the provided DataLoader, criterion and optimizer for number of epochs.
        :return: a Dict of average train and validation episode losses
        """
        self.model.to(self.device)
        self.criterion.to(self.device)
        avg_episode_losses = {"Train": [], "Validation": []}

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = torch.tensor(0.0, device=self.device)

            tqdm_dataloader = tqdm(
                self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )

            for batch in tqdm_dataloader:
                input_ids, token_type_ids, attention_mask, labels = self.get_from_batch(batch)

                outputs = self.model(
                    ids=input_ids,
                    token_type_ids={"token_type_ids": token_type_ids},
                    mask=attention_mask,
                )

                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss

            train_episode_loss = total_loss.item() / len(self.dataloader)
            if self.wandb_run:
                self.wandb_run.log({'training_loss': train_episode_loss, 'epoch': epoch})
            stop_loss = train_episode_loss
            avg_episode_losses["Train"].append(round(train_episode_loss, 4))
            results = f"Epoch {epoch + 1}/{self.num_epochs} | Loss: {train_episode_loss:.4f}"

            if self.val_dataloader:
                # Validation
                self.model.eval()  # Set the model to evaluation mode
                val_total_loss = torch.tensor(0.0, device=self.device)

                with torch.no_grad():  # Disable gradient computation during validation
                    for val_batch in self.val_dataloader:
                        (
                            val_input_ids, val_token_type_ids, val_attention_mask, val_labels
                        ) = self.get_from_batch(val_batch)

                        val_outputs = self.model(
                            ids=val_input_ids,
                            token_type_ids={"token_type_ids": val_token_type_ids},
                            mask=val_attention_mask,
                        )

                        val_loss = self.criterion(val_outputs, val_labels)
                        val_total_loss += val_loss

                val_episode_loss = val_total_loss.item() / len(self.val_dataloader)
                if self.wandb_run:
                    self.wandb_run.log({'validation_loss': val_episode_loss, 'epoch': epoch})
                stop_loss = val_episode_loss
                avg_episode_losses["Validation"].append(round(val_episode_loss, 4))
                results += f" | Val_Loss: {val_episode_loss:.4f}"

            # End of epoch
            print(results)
            if self.early_stopper.early_stop(stop_loss, epoch):
                print(f"\n--- Early stopping condition met! ---\n")
                break

        return avg_episode_losses

    def get_from_batch(self, batch) -> Tuple:
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].type(torch.BoolTensor).to(self.device)
        labels = batch["labels"].to(self.device)
        return input_ids, token_type_ids, attention_mask, labels
