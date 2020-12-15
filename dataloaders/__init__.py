from dataloaders.datasets import urfdfusion
from torch.utils.data import DataLoader
import os
import csv
import numpy as np

def load_dataset_dict(dataset='urfdfusion', split=1):
    import pandas as pd
    load_index = False
    if dataset == 'urfdfusion':
        base_path = '/home/aimaster/Desktop/DEV/vgg16-urfd-fusion/dataset/urfdTrainTestlist'
        train_list_path = os.path.join(base_path, 'trainlist{0:02d}.csv'.format(split))
        test_list_path = os.path.join(base_path, 'testlist{0:02d}.csv'.format(split))
        train_index_path = os.path.join(base_path, 'test_index.csv')
        if os.path.exists(train_index_path):
            load_index = True

    train_list_pd = pd.read_csv(train_list_path)
    val_list_pd = pd.read_csv(test_list_path)
    
    dataset_dict = {'train':{}, 'val':{}}

    dataset_dict['train']['video_name'] = train_list_pd['video_name'].to_list()
    dataset_dict['train']['num_frames'] = train_list_pd['num_frames'].to_list()
    dataset_dict['train']['class_index'] = train_list_pd['class_index'].to_list()
    dataset_dict['train']['data'] = []
    dataset_dict['train']['labels'] = []

    dataset_dict['val']['video_name'] = val_list_pd['video_name'].to_list()
    dataset_dict['val']['num_frames'] = val_list_pd['num_frames'].to_list()
    dataset_dict['val']['class_index'] = val_list_pd['class_index'].to_list()
    dataset_dict['val']['data'] = []
    dataset_dict['val']['labels'] = []

    stride = 1
    stack = 10
    modes = ['train', 'val']

    for mode in modes:
        #total_stack_frames = []
        for i, video_name in enumerate(dataset_dict[mode]['video_name']):
            num_frames = dataset_dict[mode]['num_frames'][i]
            num_stacks = int((num_frames // stride) - stack)

            for j in range(num_stacks):
                data = '{0},{1}'.format(video_name, j+1)
                dataset_dict[mode]['data'].append(data)
                dataset_dict[mode]['labels'].append(dataset_dict[mode]['class_index'][i])

    for mode in modes:
        labels_np = np.array(dataset_dict[mode]['labels'])
        stack_frames_np = np.array(dataset_dict[mode]['data'])

        if load_index == False:
            fall_indices = np.where(labels_np==1)[0]
            notfall_indices = np.where(labels_np==2)[0]
            notfall_indices_sampled = np.random.choice(notfall_indices, len(fall_indices), replace=False)
            all_indices = list(fall_indices) + list(notfall_indices_sampled)
            with open(os.path.join(base_path, '{}_index.csv'.format(mode)), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['index'])
                for index in all_indices:
                    writer.writerow([index])
        else:
            all_indices = pd.read_csv(os.path.join(base_path, '{}_index.csv'.format(mode)))['index'].to_list()
        dataset_dict[mode]['data'] = stack_frames_np[all_indices]
        dataset_dict[mode]['labels'] = labels_np[all_indices]
    
    return dataset_dict

def make_data_loader(args, **kwargs):
    if args.dataset == 'urfdfusion':
           
        dataset_dict = load_dataset_dict(dataset=args.dataset, split=1)
        
        train_set = urfdfusion.URFDFusion(dataset_dict, split='train')
        val_set = urfdfusion.URFDFusion(dataset_dict, split='val')

        num_class = 2
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

