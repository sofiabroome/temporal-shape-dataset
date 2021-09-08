import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset.ts_dataset import TemporalShapeDataset


class TemporalShapeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config: dict, seq_first: bool):
        super().__init__()
        self.data_dir = data_dir
        self.dims = (1, 64, 64)
        self.config = config
        self.seq_first = seq_first

    def setup(self, stage: str = None) -> None:
        # if stage == 'fit' or stage is None:
        if stage == 'fit':
            train_val_data = TemporalShapeDataset(
                root=self.data_dir,
                clip_size=self.config['clip_size'],
                is_val=False,
                transform_post=None,
                is_test=True,
                seq_first=self.seq_first
                )
            self.train_data, self.val_data = random_split(
                train_val_data, [self.config['nb_train_samples'], self.config['nb_val_samples']],
                generator=torch.Generator().manual_seed(42))

        # if stage == 'test' or stage is None:
        if stage == 'test':
            self.test_data = TemporalShapeDataset(
                root=self.data_dir,
                clip_size=self.config['clip_size'],
                is_val=True,
                transform_post=None,
                is_test=True,
                seq_first=self.seq_first
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False)
