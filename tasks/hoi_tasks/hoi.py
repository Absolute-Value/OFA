# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
from typing import Optional
from argparse import Namespace
from fairseq.tasks import register_task
from fairseq.data import FairseqDataset, iterators

from tasks.ofa_task import OFATask, OFAConfig
from data.hoi_data.hoi_dataset import HoiDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class HoiConfig(OFAConfig):
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )
    
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )

@register_task("hoi_task", dataclass=HoiConfig)
class HoiTask(OFATask):
    def __init__(self, cfg: HoiConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % (len(paths))]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = HoiDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
        )
        
    def build_model(self, cfg):
        model = super().build_model(cfg)
        
        return model
    
    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator