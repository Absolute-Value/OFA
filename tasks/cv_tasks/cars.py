import os
import logging
from dataclasses import dataclass, field
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.cv_data.cars_dataset import CarsDataset

logger = logging.getLogger(__name__)

@dataclass
class CarsConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )

@register_task("cars", dataclass=CarsConfig)
class CarsTask(OFATask):
    def __init__(self, cfg: CarsConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        data_dir = self.cfg.data
        if split == 'valid':
            phase = 'val'
        else:
            phase = split
        
        dataset = Dataset(data_dir, phase)

        self.datasets[split] = CarsDataset(
            phase,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            max_image_size=self.cfg.patch_image_size,
        )

class Dataset:
    def __init__(self, data_dir, split):
        self.root_dir = data_dir

        if split == 'train':
            with open(os.path.join(data_dir, f'anno_{split}.csv')) as f:
                self.data = f.read().splitlines()
            self.data = [x.split(",") for x in self.data]
        else:
            with open(os.path.join(data_dir, f'{split}_data_512.tsv')) as f:
                self.data = f.read().splitlines()
            self.data=self.data[1:]
            self.data = [x.split("\t") for x in self.data]
        
        with open(os.path.join(data_dir, f'names.csv')) as f:
            self.labels = f.read().splitlines()
        
        self.total_row_count = len(self.data)
    
    def __len__(self):
        return self.total_row_count
    
    def __getitem__(self, index):
        return self.data[index]
    
    def get_total_row_count(self):
        return self.total_row_count

    def _seek(self, offset=0):
        pass