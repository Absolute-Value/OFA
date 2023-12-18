# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import os
from fairseq.tasks import register_task
from omegaconf import DictConfig

from tasks.ofa_task import OFATask, OFAConfig
from data.hoi_data.hoi_dataset import HoiDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class HoiConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    is_multi_label: bool = field(
        default=False, metadata={"help": "whether the label is multi-label"}
    )

@register_task("hoi_task", dataclass=HoiConfig)
class HoiTask(OFATask):
    def __init__(self, cfg: HoiConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        
        dataset = Dataset(file_path)

        self.datasets[split] = HoiDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            max_image_size=self.cfg.max_image_size,
            is_multi_label=self.cfg.is_multi_label,
        )

class Dataset:
    def __init__(self, file_path):
        self.root_dir = os.path.dirname(file_path)
        with open(file_path) as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            img_id, img_name, anns = line.rstrip('\n').split("\t")
            for ann in anns.split("&&"):
                self.data.append([img_id, img_name, ann])
                if "hico-det" in file_path:
                    break
        self.total_row_count = len(self.data)
    
    def __len__(self):
        return self.total_row_count
    
    def __getitem__(self, index):
        return self.data[index]
    
    def get_total_row_count(self):
        return self.total_row_count

    def _seek(self, offset=0):
        pass