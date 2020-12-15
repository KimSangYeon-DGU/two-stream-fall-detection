import os
import csv
import math
import random

base_path = '/home/aimaster/Downloads/URFD/URFD_opticalflow'

with open('datalist.txt', 'r') as f:
    lines = f.readlines()
    fall_dir_names = []
    notfall_dir_names = []

    for line in lines:
        if line.startswith('Falls'):
            fall_dir_names.append(line.strip())
        elif line.startswith('NotFalls'): 
            notfall_dir_names.append(line.strip())

    num_fall_train = round((len(fall_dir_names) * 0.8) + 0.5)
    num_notfall_train = round((len(notfall_dir_names) * 0.8) + 0.5)

    random.shuffle(fall_dir_names)
    random.shuffle(notfall_dir_names)

    fall_train_dir_names = fall_dir_names[:num_fall_train]
    fall_test_dir_names = fall_dir_names[num_fall_train:]

    notfall_train_dir_names = notfall_dir_names[:num_notfall_train]
    notfall_test_dir_names = notfall_dir_names[num_notfall_train:]

    train_dir_names = fall_train_dir_names + notfall_train_dir_names
    test_dir_names = fall_test_dir_names + notfall_test_dir_names

    print(len(train_dir_names))
    print(len(test_dir_names))
    
    with open('trainlist{0:02d}.csv'.format(1), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['video_name', 'num_frames', 'class_index'])
        for dir_name in train_dir_names:
            name = dir_name.split(' ')[0]
            label = int(dir_name.split(' ')[1])
            num_frames = len(os.listdir(os.path.join(base_path, name))) / 2
            if num_frames >= 10:
                writer.writerow([name, num_frames, label])    

    with open('testlist{0:02d}.csv'.format(1), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['video_name', 'num_frames', 'class_index'])
        for dir_name in test_dir_names:
            name = dir_name.split(' ')[0]
            label = int(dir_name.split(' ')[1])
            num_frames = len(os.listdir(os.path.join(base_path, name))) / 2
            if num_frames >= 10:
                writer.writerow([name, num_frames, label])    
