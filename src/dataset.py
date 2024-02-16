from typing import Dict, List, NamedTuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class ProteinData(NamedTuple):
    ligand: str
    protein: str
    interaction_metric: int  # IC50 or K_d


class ProteinSMILESDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ic50_metric=True, kd_metric=False) -> None:
        assert ic50_metric or kd_metric, "Metric must be either IC50 or Kd values, can't be both or neither"
        self.df = df
        self.ic50_metric = ic50_metric

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ProteinData:
        row = self.df.iloc[idx]

        ligand = row["Ligand SMILES"]
        protein = row["BindingDB Target Chain Sequence"]

        # Calculate p_metric
        if self.ic50_metric:
            target_ic50 = row["IC50 (nM)"]
            target = 9 - np.log10(target_ic50)
        # Calculate pKd
        else:
            target_kd = row["Kd (nM)"]
            target = 9 - np.log10(target_kd)

        data = ProteinData(ligand, protein, target)

        return data


class ProteinSMILESDavisDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ProteinData:
        row = self.df.iloc[idx]

        ligand = row["SMILES"]
        protein = row["Sequence"]
        target = row["Kd"]

        data = ProteinData(ligand, protein, target)

        return data


class TransformerCollate:
    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def __call__(self, batches: List[ProteinData]) -> Dict[str, torch.Tensor]:
        ligands = [item.ligand for item in batches]
        proteins = [item.protein for item in batches]
        target = [item.interaction_metric for item in batches]

        encodings = self.tokenizer(
            ligands, proteins, padding=True, truncation=True, return_tensors="pt"
        )

        encodings["labels"] = torch.tensor(target, dtype=torch.float32).view(-1, 1)

        return encodings


def filter_outlier_ic50(data: pd.DataFrame, threshold: float | int) -> pd.DataFrame:
    mask = data["IC50 (nM)"] > threshold
    data = data[~mask]
    return data


def main() -> None:
    df = pd.read_csv("../BindingDB_EQ_IC50_Subset.tsv", sep="\t")
    df = filter_outlier_ic50(df, 10e5)
    df.to_csv('../BindingDB_EQ_IC50_Subset_filtered.tsv', sep="\t", index=False)

    dataset = ProteinSMILESDataset(df)

    collate_fn = TransformerCollate("../Chem_Tokenizer")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(dataset[0])

    batch = next(iter(dataloader))

    print(batch["input_ids"].shape)
    print(batch["token_type_ids"].tolist())


if __name__ == "__main__":
    main()
