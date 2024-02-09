from src.model import IC50Bert
import torch
from typing import List
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class IC50BertTrainer:
    """
    Class used in the training of an IC50Bert model
    """

    def __init__(
        self,
        model: IC50Bert,
        dataloader: DataLoader,
        num_epochs: int,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device("cuda")
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopper = EarlyStopper(patience=10, min_delta=0.05)

    def train(self) -> List:
        """
        Train the specified model using the provided DataLoader, criterion and optimizer for number of epochs.
        :return: a List of average episode losses
        """
        self.model.to(self.device)
        self.criterion.to(self.device)
        avg_episode_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            tqdm_dataloader = tqdm(
                self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )

            for batch in tqdm_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].type(torch.BoolTensor).to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    ids=input_ids,
                    token_type_ids={"token_type_ids": token_type_ids},
                    mask=attention_mask,
                )

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                tqdm_dataloader.set_postfix(loss=loss.item())

            episode_loss = total_loss / len(self.dataloader)
            avg_episode_losses.append(round(episode_loss, 4))
            tqdm_dataloader.close()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {episode_loss:.4f}")
            # if self.early_stopper.early_stop(episode_loss):
            #     print(f"\n--- Early stopping condition met! ---\n")
            #     break

        return avg_episode_losses
