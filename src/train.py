from src.model import IC50Bert
import torch
from typing import List
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


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

            average_loss = total_loss / len(self.dataloader)
            avg_episode_losses.append(round(average_loss, 4))
            tqdm_dataloader.close()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {average_loss:.4f}")

        return avg_episode_losses
