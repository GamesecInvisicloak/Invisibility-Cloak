import random
import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

def load_dataset(game,test_batch_size=1,patch=None,scenario=None,shuffle=False,train_split=0.5,train=True,demo=False):
    if demo:
        dataset = GameDatasetDemo(game=game,dataset_root='data',action=scenario)
    else:
        dataset = GameDataset(game=game,dataset_root='data',scenario=scenario,patch=patch,train_split=train_split,train=train)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False,num_workers=4)
    return dataloader

class GameDataset(Dataset):
    def __init__(self, game, dataset_root,scenario='dust2',patch=0,train_split=0.5,train=False):
        super(GameDataset, self).__init__()
        self.images = []
        self.boxes = []
        self.locs = []

        # all scenario
        if scenario is None and patch is None:
            scenario_list = os.listdir(os.path.join(dataset_root,game))
            for scenario in scenario_list:
                if 'train' in scenario or'val' in scenario:
                    continue
                patch_list = os.listdir(os.path.join(dataset_root,game, scenario,'clean'))
                sorted_patch_list = sorted(patch_list, key=lambda x: int(os.path.basename(x)))
                random.seed(0)
                random.shuffle(sorted_patch_list)
                if train:
                    selected_list = sorted_patch_list[:int(len(sorted_patch_list)*train_split)]
                    # print(selected_list)
                else:
                    selected_list = sorted_patch_list[int(len(sorted_patch_list)*train_split):]
                    # print(selected_list)
                for p in selected_list:
                    # load videos
                    clean_output_dir = os.path.join(dataset_root,game, scenario, 'clean',str(p))
                    json_output_dir = os.path.join(dataset_root,game, scenario,'json',str(p))
                    # sort by time
                    original_files = os.listdir(clean_output_dir)
                    num_list = [int(f.split('_')[1].split('.')[0]) for f in original_files ]
                    sorted_files = [f for _, f in sorted(zip(num_list, original_files))]
                    self.images.extend([os.path.join(clean_output_dir, os.path.splitext(file)[0] + '.jpg') for file in sorted_files])
                    self.locs.extend([(scenario,p) for file in sorted_files])
                    # self.boxes.extend([os.path.join(json_output_dir, os.path.splitext(file)[0] + '.json') for file in sorted_files])
        else:  # choose scenario
            if patch is None:
                patch_list = os.listdir(os.path.join(dataset_root,game, scenario,'clean'))
                sorted_patch_list = sorted(patch_list, key=lambda x: int(os.path.basename(x)))
                random.seed(0)
                random.shuffle(sorted_patch_list)
                if train:
                    selected_list = sorted_patch_list[:int(len(sorted_patch_list)*train_split)]
                else:
                    selected_list = sorted_patch_list[int(len(sorted_patch_list)*train_split):]

                for p in selected_list:
                    # load videos
                    clean_output_dir = os.path.join(dataset_root,game, scenario, 'clean',str(p))
                    json_output_dir = os.path.join(dataset_root,game, scenario,'json',str(p))
                    # sort by time
                    original_files = os.listdir(clean_output_dir)
                    num_list = [int(f.split('_')[1].split('.')[0]) for f in original_files ]
                    sorted_files = [f for _, f in sorted(zip(num_list, original_files))]
                    self.images.extend([os.path.join(clean_output_dir, os.path.splitext(file)[0] + '.jpg') for file in sorted_files])
                    self.locs.extend([(scenario,p) for file in sorted_files])
                    # self.boxes.extend([os.path.join(json_output_dir, os.path.splitext(file)[0] + '.json') for file in sorted_files])


        if (not scenario is None) and (not patch is None):
            # load videos
            clean_output_dir = os.path.join(dataset_root,game, scenario, 'clean',str(patch))
            json_output_dir = os.path.join(dataset_root,game, scenario,'json',str(patch))
            # sort by time
            original_files = os.listdir(clean_output_dir)
            num_list = [int(f.split('_')[1].split('.')[0]) for f in original_files ]
            sorted_files = [f for _, f in sorted(zip(num_list, original_files))]
            self.images = [os.path.join(clean_output_dir, os.path.splitext(file)[0] + '.jpg') for file in sorted_files]
            self.locs.extend([(scenario,p) for file in sorted_files])
            # self.boxes = [os.path.join(json_output_dir, os.path.splitext(file)[0] + '.json') for file in sorted_files]

        print('len dataset:',len(self.images))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.images[index]
        (scenario,p) = self.locs[index]
        return img_path,scenario,p

    def __len__(self) -> int:
        return len(self.images)

    
class GameDatasetDemo(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016', 'VOT2019'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, game, dataset_root,action='dust2_run'):
        super(GameDatasetDemo, self).__init__()
        dataset_root = os.path.join(dataset_root,game)
        clean_output_dir = os.path.join(dataset_root, action, 'clean')
        # sort by time
        original_files = os.listdir(clean_output_dir)
        num_list = [int(f.split('_')[1].split('.')[0]) for f in original_files ]

        sorted_files = [f for _, f in sorted(zip(num_list, original_files))]
        self.images = [os.path.join(clean_output_dir, os.path.splitext(file)[0] + '.jpg') for file in sorted_files]

        print(action, len(self.images))

    def __getitem__(self, index: int):
        img_path = self.images[index]
        return img_path,0,0

    def __len__(self) -> int:
        return len(self.images)
    