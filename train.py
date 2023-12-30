from model import IC50Bert
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class IC50BertTrainer:
    """
    Class used in the training of an IC50Bert model
    """
    def __init__(self, model: IC50Bert, dataloader: DataLoader, num_epochs: int,
                 criterion: nn.Module, optimizer: optim.Optimizer) -> None:
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self) -> None:
        """
        Train the specified model using the provided DataLoader, criterion and optimizer for number of epochs.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            tqdm_dataloader = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in tqdm_dataloader:
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                attention_mask = batch['attention_mask'].type(torch.BoolTensor)
                labels = batch['labels'].view(-1, 1).type(torch.float32)

                self.optimizer.zero_grad()

                outputs = self.model(
                    ids=input_ids,
                    token_type_ids={"token_type_ids": token_type_ids},
                    mask=attention_mask
                )

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                tqdm_dataloader.set_postfix(loss=loss.item())

            average_loss = total_loss / len(self.dataloader)
            tqdm_dataloader.set_postfix(avg_loss=average_loss)
            tqdm_dataloader.close()
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {average_loss:.4f}')
