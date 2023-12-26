import os
import numpy as np
import torch
import argparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.hoi_tasks.hoi_task import HoiTask

import matplotlib.pyplot as plt
from PIL import Image

tasks.register_task('hoi', HoiTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = True

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--model_size', type=str, default='tiny')
parser.add_argument('--dataset_name', type=str, default='hico')
args = parser.parse_args()
img_size = args.img_size
model_size = args.model_size
dataset_name = args.dataset_name

# specify some options for evaluation
parser = options.get_generation_parser()
# input_args = ["", "--task=refcoco", "--beam=10", "--path=checkpoints/ofa_large.pt", "--bpe-dir=utils/BPE", "--no-repeat-ngram-size=3", "--patch-image-size=384"]
input_args = ["", "--task=hoi_task", "--beam=10", f"--path=run_scripts/hoi/hoi_checkpoints/{dataset_name}/{model_size}/30_1000_5e-5_{img_size}/checkpoint_best.pt", "--bpe-dir=utils/BPE", "--no-repeat-ngram-size=3", f"--patch-image-size={img_size}"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# configファイルと学習済みモデルのロード
task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path),
    task=task
)
print(len(models))

# GPUに載せる
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# generatorの初期化
generator = task.build_generator(models, cfg.generation)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")

# Image transform
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
        else:
          token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int((coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int((coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip()))  # 下の条件以外を単語ごとにencodeして数字にし、配列にぶち込む
      # BPE: Byte Pair Encodingの頭文字であり、文書における低頻度の単語をさらに分割することで、低頻度の単語もうまく扱えるようにする手法
      if not word.startswith('<code_') and not word.startswith('<bin_') else word # <code_ と <bin_ はそのまま通す
      for word in text.strip().split() # 両端の空白を削除し、空白で分割（単語に）
    ]

    line = ' '.join(line) # 配列をつなげる
    s = task.tgt_dict.encode_line( # fairseq/data/dictionary.py
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long() # タスクごとに文章をエンコード　数値が変わっている　トークン化？
    
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s]) # スタート[0]をつける
    if append_eos:
        s = torch.cat([s, eos_item]) # エンド[2]をつける
    return s

def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0) # lower()で全て小文字にし、strip()で両端の空白を削除→encode_textで文字をベクトルにエンコード
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample
  
def apply_half(t): # Function to turn FP32 to FP16
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

offset = 0
file_dir = f'/local/{dataset_name}/'
fp = open(os.path.join(file_dir, 'test_ofa.tsv'), "r")
lineid_to_offset = []
for line in fp:
    lineid_to_offset.append(offset)
    offset += len(line.encode('utf-8'))

with open(f'/data01/{dataset_name}/hoi_classes.txt', 'r') as f:
    hoi_classes = f.read().splitlines()

def HOI(img_number=0, save_dir=False, is_print=True):
    fp.seek(lineid_to_offset[img_number])
    image_id, ori_image_path, label = fp.readline().rstrip("\n").split("\t")
    image_path = os.path.join(file_dir, ori_image_path)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    boxes_target = {"human_boxes": [], "obj_boxes": [], "hoi_ids": [], "hois": [], "objs": [], "human_area": [], "obj_area": [], "size": torch.tensor([h, w])}
    label_list = label.strip().split('&&')
    for label in label_list:
        human_x0, human_y0, human_x1, human_y1, hoi_id, hoi, obj_x0, obj_y0, obj_x1, obj_y1, obj_id, obj = label.strip().split(',', 12)
        boxes_target["human_boxes"].append([float(human_x0), float(human_y0), float(human_x1), float(human_y1)])
        boxes_target["obj_boxes"].append([float(obj_x0), float(obj_y0), float(obj_x1), float(obj_y1)])
        boxes_target["hoi_ids"].append(hoi_id)
        boxes_target["hois"].append(hoi)
        boxes_target["objs"].append(obj)
    human = " ".join(["<bin_{}>".format(int(pos)) for pos in boxes_target["human_boxes"][0][:4]])
    obj_name = boxes_target["objs"][0]
    obj = " ".join(["<bin_{}>".format(int(pos)) for pos in boxes_target["obj_boxes"][0][:4]])
    hoi_ids = " ".join(boxes_target["hoi_ids"][0].split('%'))
    instruction = f'What are the interaction between person {human} and {obj_name} {obj}?'
    hoi_names = " ".join(boxes_target["hois"][0].split('%'))
    sample = construct_sample(image, instruction)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        # print(hypos[0][0]["tokens"])
        tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
    
    return ori_image_path, instruction, hoi_ids, tokens

def string_to_int(s):
    return list(map(int, s.split()))

result = []
trues = []
preds = []
write_str = 'img_path\tsrc\tans\tpred\n'
for i in range(len(lineid_to_offset)):
    img_path, src, ans, pred = HOI(i, is_print=False)
    result.append(ans==pred)
    trues.append(string_to_int(ans))
    preds.append(string_to_int(pred))
    write_str += f'{img_path}\t{src}\t{ans}\t{pred}\n'
# MultiLabelBinarizerのインスタンスを作成
mlb = MultiLabelBinarizer()
trues_binarized = mlb.fit_transform(trues)
preds_binarized = mlb.transform(preds)
micro_f1 = f1_score(trues_binarized, preds_binarized, average='micro')*100
macro_f1 = f1_score(trues_binarized, preds_binarized, average='macro')*100
output_dir = f'results/{dataset_name}/{model_size}/'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, f'{img_size}_results.tsv'), 'w') as f:
    f.write(write_str)
acc = sum(result)/len(result)*100
with open(os.path.join(output_dir, f'{img_size}_result.txt'), 'w') as f:
    f.write(f'Total pram:{total_params}\nTrain pram:{trainable_params}\nAcc:{round(acc, 2)} ({acc})\nMicro F1:{round(micro_f1, 2)} ({micro_f1})\nMacro F1:{round(macro_f1, 2)} ({macro_f1})')
print(dataset_name, model_size, img_size, round(acc, 2))
print(f'Micro F1:{round(micro_f1, 2)}\nMacro F1:{round(macro_f1, 2)}')