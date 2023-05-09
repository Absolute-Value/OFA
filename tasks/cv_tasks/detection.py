from dataclasses import dataclass, field
import logging
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.cv_data.detection_dataset import DetectionDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )

@register_task("detection_task", dataclass=DetectionConfig)
class DetectionTask(OFATask):
    def __init__(self, cfg: DetectionConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = DetectionDataset(
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
        )