import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.class2index, self.index2class = self.load_class_info()
    
    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, images, targets, preds, global_step):
        nth = np.random.randint(0, targets.shape[0])
        frames = torch.zeros((20, 1, 224, 224))
        for i in range(images.size(1)):
            a = images[nth, i,...]
            frames[i,...] = a.unsqueeze(0)
        grid = make_grid(frames, nrow=5)

        preds = (preds >= 0.5).astype(int)
        targets = (targets >= 0.5).astype(int)

        target_class = self.to_class(targets[nth] + 1)
        pred_class = self.to_class(preds[nth] + 1)
        
        writer.add_figure('Optical Flows', self.to_figure(grid, target_class, pred_class), global_step)

    def get_top_class(self, preds):
        return np.argsort(preds, axis=1)[:, -1:]

    def to_figure(self, grid, target, pred):
        fig = plt.figure(figsize=(10, 10))
        #np_img = grid.cpu().numpy()
        plt.xlabel('Target: {0}\nPredicted: {1}'.format(target, pred), fontsize=24)
        grid = 0.5 + grid * 0.226
        plt.imshow(grid.permute(1, 2, 0))
        #plt.imshow(np.transpose((np_img * 255).astype('uint8'), (1, 2, 0)), interpolation='nearest')
        return fig

    def to_class(self, index):
        return self.index2class[str(index)]

    def load_class_info(self):
        class2index = {}
        index2class = {}
        with open('dataset/urfdTrainTestlist/classInd.txt', 'r') as f:
            for l in f.readlines():
                idx, class_name = l.split(' ')[0], l.split(' ')[1]
                class2index[class_name.strip()] = idx
                index2class[str(idx)] = class_name
        
        return class2index, index2class
