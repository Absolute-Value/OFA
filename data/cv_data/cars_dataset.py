# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os

import logging
import warnings
import string
import copy
import glob
import random

import numpy as np
import torch

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


class CarsDataset(OFADataset):
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
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        
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
    
    def get_img_path(self,dir_name):
        """
        指定されたディレクト以下の画像パスから一つランダムに返す"""
        img_paths = glob.glob(dir_name + '/*.jpg')
        return random.choice(img_paths)  
    
    def train_data_proc(self,d):
        label_id = int(d[5]) - 1
        label = self.dataset.labels[label_id]
        
        #対象ラベルを省いて残りから3つ選ぶ
        sub_label_list = copy.deepcopy(self.dataset.labels)
        sub_label_list.remove(label)
        sub_labels = random.sample(sub_label_list,3)

        #画像読み込み
        main_img_path = os.path.join(self.dataset.root_dir, "car_data/car_data", self.split, label.replace("/", "-"),d[0])
        sub_img_paths = [self.get_img_path(os.path.join(self.dataset.root_dir, "car_data/car_data", self.split, sub_label.replace("/","-"))) for sub_label in sub_labels]
        main_img = Image.open(main_img_path).convert('RGB')
        main_w, main_h = main_img.size
        main_img = np.array(main_img.resize((128,128)))
        imgs = [main_img]
        for sub_img_path in sub_img_paths:
            _img = np.array(Image.open(sub_img_path).convert('RGB').resize((128,128)))
            imgs.append(_img)

        #画像合成
        img = np.zeros((256,256,3))
        id_list = [0,1,2,3] #0がmain   
        random.shuffle(id_list)
        img[:128,:128,:] = imgs[id_list[0]]
        img[:128,128:,:] = imgs[id_list[1]]
        img[128:,:128,:] = imgs[id_list[2]]
        img[128:,128:,:] = imgs[id_list[3]]
        img = Image.fromarray(np.uint8(img))
        #loc作成
        offset_dict = [[0,0],[0.55,0],[0,0.55],[0.55,0.55]] #40で区切る関係で中心が0.5にならない。0.55は暫定値
        x1,y1,x2,y2 = d[1:5]

        #存在位置によるオフセット処理
        x1 = float(x1)/float(2.0*main_w) +offset_dict[id_list.index(0)][0]  
        y1 = float(y1)/float(2.0*main_h) +offset_dict[id_list.index(0)][1]
        x2 = float(x2)/float(2.0*main_w) +offset_dict[id_list.index(0)][0]
        y2 = float(y2)/float(2.0*main_h) +offset_dict[id_list.index(0)][1]
        
        if x2>1.0:
            x2 = 1.0
        if y2>1.0:
            y2 = 1.0
        locs = [x1,y1,x2,y2]
        
        return locs, img, label_id
        

    def __getitem__(self, index):
        d = self.dataset[index] #d[0]:画像名 d[1:5]:bbox d[5]:label
        if self.split == "train":
            locs,image,label_id = self.train_data_proc(d)
        else:
            locs = [float(l) for l in d[3].split("&&")]
            image = Image.open(os.path.join(self.dataset.root_dir, f"{self.split}_img_512",d[0])).convert('RGB')
            label_id = int(d[1])
        
        w, h = image.size
        boxes_target = {"obj_boxes": [], "size": torch.tensor([h, w])}
        boxes_target["obj_boxes"].append(locs)
        boxes_target["obj_boxes"] = torch.tensor(boxes_target["obj_boxes"])

        patch_image, patch_boxes = self.detection_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])

        for obj_box in patch_boxes["obj_boxes"]:
            quant_boxes = []
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in obj_box[:4]])
        src_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)
        tgt_item = self.encode_text(str(label_id))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": index,
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
