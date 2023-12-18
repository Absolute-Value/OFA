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
    
    w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
    h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)
    # region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

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
        "w_resize_ratios": w_resize_ratios,
        "h_resize_ratios": h_resize_ratios,
        # "region_coords": region_coords
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
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512,
        is_multi_label=False,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.is_multi_label = is_multi_label
        
        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        self.detection_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
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
            human_x0, human_y0, human_x1, human_y1, hoi_id, hoi, obj_x0, obj_y0, obj_x1, obj_y1, obj_id, obj = label.strip().split(',')
            boxes_target["human_boxes"].append([float(human_x0), float(human_y0), float(human_x1), float(human_y1)])
            boxes_target["obj_boxes"].append([float(obj_x0), float(obj_y0), float(obj_x1), float(obj_y1)])
            if self.is_multi_label:
                boxes_target["hois"].append(" ".join(hoi_id.split("%")))
            else:
                boxes_target["hois"].append(hoi_id)
            boxes_target["objs"].append(obj)
            boxes_target["human_area"].append((float(human_x1) - float(human_x0)) * (float(human_y1) - float(human_y0)))
            boxes_target["obj_area"].append((float(obj_x1) - float(obj_x0)) * (float(obj_y1) - float(obj_y0)))
        boxes_target["human_boxes"] = torch.tensor(boxes_target["human_boxes"])
        boxes_target["obj_boxes"] = torch.tensor(boxes_target["obj_boxes"])
        boxes_target["hois"] = np.array(boxes_target["hois"])
        boxes_target["objs"] = np.array(boxes_target["objs"])
        boxes_target["human_area"] = torch.tensor(boxes_target["human_area"])
        boxes_target["obj_area"] = torch.tensor(boxes_target["obj_area"])

        patch_image, patch_boxes = self.detection_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])

        for i, (human_box, obj_box) in enumerate(zip(patch_boxes["human_boxes"], patch_boxes["obj_boxes"])):
            quant_boxes = []
            quant_boxes.extend(self.bpe.encode('What is the interaction between person'))
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in human_box[:4]])
            quant_boxes.append(self.bpe.encode(' and {}'.format(boxes_target["objs"][i])))
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in obj_box[:4]])
            quant_boxes.append(self.bpe.encode('?'))
            tgt_text = boxes_target["hois"][i]
            break
        src_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)
        tgt_item = self.encode_text(tgt_text)

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
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
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
