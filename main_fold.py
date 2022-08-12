import argparse
import os
import shutil

from datetime import datetime
from torch.utils.data import DataLoader

from datasets.dataset_3d import SegDataset
from nets.segment_model_split import DPModel
import torchutil
import torch
import time

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from datasets.DataLoaderX import DataLoaderX

parser = argparse.ArgumentParser(description='main')

parser.add_argument("--model", help="network model")
parser.add_argument('--gpu', default='0', type=str,
                    help='use gpu device.')

parser.add_argument('--config', default='cfgs/default.yaml', type=str,
                    help='configuration file. default=cfgs/default.yaml')
parser.add_argument('--keep_run_kfold', default=0, type=int,
                    help='keep_run_kfold')
parser.add_argument('--run_timestamp', default='', type=str,
                    help='run_timestamp.')

args = parser.parse_args()
config = torchutil.parse.parse_yaml(args.config)
start_fold = args.keep_run_kfold
if start_fold > 0:
    run_timestamp = args.run_timestamp
else:
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
logging_params = config['logging']
task = config['task']
ckpt_path = os.path.join(logging_params['ckpt_path'], f'{task}/{args.model}')
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
ckpt_path = os.path.join(ckpt_path, run_timestamp)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
logging_params['ckpt_path'] = ckpt_path
if not os.path.exists(os.path.join(ckpt_path, "datasets")):
    shutil.copytree("./datasets", os.path.join(ckpt_path, "datasets"))
    shutil.copytree("./nets", os.path.join(ckpt_path, "nets"))
    shutil.copy(args.config, os.path.join(ckpt_path, 'config.yaml'))

num_gpus = torchutil.gpu.set_gpu(args.gpu)
network_params = config['network']
network_params['model_name'] = args.model
network_params['seed'] = config['seed']
network_params['device'] = "cuda" if num_gpus > 0 else "cpu"
network_params['use_parallel'] = num_gpus > 1  # str2bool(args.use_parallel)
network_params['num_gpus'] = num_gpus
network_params['spacing'] = list(config['data']['spacing'])
network_params['patch_size'] = list(config['data']['patch_size'])


def collect_fn(batch):
    image = torch.cat([item["image"] for item in batch], dim=0)
    target = torch.cat([item["label"] for item in batch], dim=0).float()
    return {"image": image, "label": target}


# load data
data_params = config['data']
eval_params = config['eval']
train_params = config['train']

pretrained_path = config['network']['pretrained_path']

for split in range(start_fold, 5):
    logging_params['ckpt_path'] = os.path.join(ckpt_path, "split{}".format(split))
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    config['network']['pretrained_path'] = os.path.join(pretrained_path, f"split{split}/best.pth")
    
    data_params['eval_sample_csv'] = os.path.join(data_params['sample_csv_root'], 
                                                  f"fold_{split}/test.txt")
    data_params['train_sample_csv'] = os.path.join(data_params['sample_csv_root'],
                                                   f"fold_{split}/train.txt")

    eval_params = config['eval']
    #eval_trans_seq = resolve_transforms(eval_params['aug_trans'])
    eval_dataset = SegDataset(
        data_root=data_params['data_root'],
        sample_txt=data_params['eval_sample_csv'],
        dataset_info=data_params['dataset_info'],
        num_works=eval_params['num_workers'],
        #transforms=eval_trans_seq,
        patch_size=data_params['patch_size'],
        patch_overlap=data_params['patch_overlap'],
        batch_size=eval_params['batch_size'],
        #num_patch=eval_params['samples_per_volume'],
        normalization=data_params['normalization'],
        trans=eval_params['trans'],
        split="eval"
    )

    train_params = config['train']
    # train_trans_seq = resolve_transforms(train_params['aug_trans'])
    train_dataset = SegDataset(
        data_root=data_params['data_root'],
        sample_txt=data_params['train_sample_csv'],
        dataset_info=data_params['dataset_info'],
        num_works=train_params['num_workers'],
        batch_size=train_params['batch_size'],
        # transforms=train_trans_seq,
        patch_size=data_params['patch_size'],
        samples_per_volume=train_params['samples_per_volume'],
        num_iterations = train_params['num_iterations'],
        normalization=data_params['normalization'],
        trans=train_params['trans'],
        split="train")
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        # num_workers=train_params['num_workers'],
        # drop_last=True,
        # pin_memory=train_params['pin_memory'],
        # collate_fn=collect_fn
    )

    net = DPModel(config=config)

    torch.backends.cudnn.benchmark = True

    net.train(train_loader, eval_dataset)
    del eval_dataset
    del train_dataset
    del train_loader
    del net
    torch.cuda.empty_cache()
    print('please wait 10s....')
    time.sleep(10)
    torch.cuda.empty_cache()
