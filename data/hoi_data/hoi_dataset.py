# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
import utils.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


class HoiDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        num_bins=1000,
        patch_image_size=224,
        code_image_size=128,
        max_image_size=512,
        imagenet_default_mean_and_std=False,
        scst=False
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size
        self.scst = scst

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

    def __getitem__(self, index):
        image_id, image, label = self.dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")

        w, h = image.size
        boxes_target = {"human_boxes": [], "obj_boxes": [], "hois": [], "objs": [], "human_area": [], "obj_area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label in label_list:
            # x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            human_x0, human_y0, human_x1, human_y1, hoi_id, hoi, obj_x0, obj_y0, obj_x1, obj_y1, obj_id, obj = label.strip().split(',', 12)
            boxes_target["human_boxes"].append([float(human_x0), float(human_y0), float(human_x1), float(human_y1)])
            boxes_target["obj_boxes"].append([float(obj_x0), float(obj_y0), float(obj_x1), float(obj_y1)])
            boxes_target["hois"].append(hoi)
            boxes_target["objs"].append(obj)
            boxes_target["human_area"].append((float(human_x1) - float(human_x0)) * (float(human_y1) - float(human_y0)))
            boxes_target["obj_area"].append((float(obj_x1) - float(obj_x0)) * (float(obj_y1) - float(obj_y0)))
        boxes_target["human_boxes"] = torch.tensor(boxes_target["human_boxes"])
        boxes_target["obj_boxes"] = torch.tensor(boxes_target["obj_boxes"])
        boxes_target["hois"] = np.array(boxes_target["hois"])
        boxes_target["objs"] = np.array(boxes_target["objs"])
        boxes_target["human_area"] = torch.tensor(boxes_target["human_area"])
        boxes_target["obj_area"] = torch.tensor(boxes_target["obj_area"])

        patch_image, boxes_target = self.patch_resize_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, (human_box, obj_box) in enumerate(zip(boxes_target["human_boxes"], boxes_target["obj_boxes"])):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in human_box[:4]])
            quant_boxes.append(self.bpe.encode(' {}'.format(boxes_target["hois"][i])))
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in obj_box[:4]])
            quant_boxes.append(self.bpe.encode(' {}'.format(boxes_target["objs"][i])))
        src_item = self.encode_text(' what are the interactions in the image?')
        tgt_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
