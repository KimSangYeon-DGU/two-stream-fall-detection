import argparse
import os
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.vgg16 import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(
            args, **kwargs)

        # Define weight
        self.temporal_weight = args.temporal_weight
        self.spatial_weight = args.spatial_weight

        # Define network
        temporal_model = Model(name='vgg16_bn', num_classes=101, is_flow=True).get_model()
        spatial_model = Model(name='vgg16_bn', num_classes=101, is_flow=False).get_model()

        # Define Optimizer
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        temporal_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=args.temporal_lr)
        spatial_optimizer = torch.optim.Adam(spatial_model.parameters(), lr=args.spatial_lr)

        # Define Criterion
        self.temporal_criterion = nn.BCELoss().cuda()
        self.spatial_criterion = nn.BCELoss().cuda()

        self.temporal_model, self.temporal_optimizer = temporal_model, temporal_optimizer
        self.spatial_model, self.spatial_optimizer = spatial_model, spatial_optimizer

        # Define Evaluator
        self.top1_eval = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.temporal_model = torch.nn.DataParallel(
                self.temporal_model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.temporal_model)
            self.temporal_model = self.temporal_model.cuda()

            self.spatial_model = torch.nn.DataParallel(
                self.spatial_model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.spatial_model)
            self.spatial_model = self.spatial_model.cuda()

        # Resuming checkpoint
        self.best_accuracy = 0.0

        '''
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
                #self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            #self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_accuracy = checkpoint['best_accuracy']
            print("=> loaded checkpoint '{}' (epoch {}), best prediction {}"
                  .format(args.resume, checkpoint['epoch'], self.best_accuracy))
        '''

    def training(self, epoch):
        train_loss = 0.0
        self.temporal_model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            rgbs, flows, targets = sample['rgb'], sample['flow'], sample['label']
            targets = targets.view(-1, 1).float()
            if self.args.cuda:
                rgbs, flows, targets = rgbs.cuda(), flows.cuda(), targets.cuda()

            self.temporal_optimizer.zero_grad()
            self.spatial_optimizer.zero_grad()

            temporal_output = self.temporal_model(flows)
            spatial_output = self.spatial_model(rgbs)

            temporal_loss = self.temporal_criterion(temporal_output, targets)
            spatial_loss = self.spatial_criterion(spatial_output, targets)

            temporal_loss.backward()
            spatial_loss.backward()

            self.temporal_optimizer.step()
            self.spatial_optimizer.step()

            train_loss += temporal_loss.item()
            train_loss += spatial_loss.item()
            
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar(
                'train/total_temporal_loss_iter', temporal_loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar(
                'train/total_spatial_loss_iter', spatial_loss.item(), i + num_img_tr * epoch)
            
            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
            #    self.summary.visualize_image(self.writer, images, targets.squeeze(1).cpu().numpy(), output.squeeze(1).data.cpu().numpy(), global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, i * self.args.batch_size + rgbs.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.temporal_model.eval()
        self.spatial_model.eval()

        self.top1_eval.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            rgbs, flows, targets = sample['rgb'], sample['flow'], sample['label']
            targets = targets.view(-1, 1).float()
            if self.args.cuda:
                rgbs, flows, targets = rgbs.cuda(), flows.cuda(), targets.cuda()
            with torch.no_grad():
                temporal_output = self.temporal_model(flows)
                spatial_output = self.spatial_model(rgbs)

            temporal_loss = self.temporal_criterion(temporal_output, targets)
            spatial_loss = self.spatial_criterion(spatial_output, targets)

            test_loss += temporal_loss.item()
            test_loss += spatial_loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            pred = temporal_output.data.cpu().numpy() * self.temporal_weight + spatial_output.data.cpu().numpy() * self.spatial_weight
            targets = targets.cpu().numpy()
            # Add batch sample into evaluator
            self.top1_eval.add_batch(targets, pred)

        # Fast test during the training
        top1_acc = self.top1_eval.Accuracy()

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/Acc', top1_acc, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' %
              (epoch, i * self.args.batch_size + rgbs.data.shape[0]))
        print("Top1: acc:{}, best accuracy:{}".format(top1_acc, self.best_accuracy))
        print("Sensitivity:{}, Specificity:{}".format(self.top1_eval.Sensitivity(), self.top1_eval.Specificity()))
        print("Confusion Maxtrix:\n{}".format(self.top1_eval.Confusion_Matrix()))
        print('Loss: %.3f' % test_loss)

        if top1_acc > self.best_accuracy:
            is_best = True
            self.best_accuracy = top1_acc
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'temporal_state_dict': self.temporal_model.module.state_dict(),
                'temporal_optimizer': self.temporal_optimizer.state_dict(),
                'spatial_state_dict': self.spatial_model.module.state_dict(),
                'spatial_optimizer': self.spatial_optimizer.state_dict(),
                'best_accuracy': self.best_accuracy,
                'sensitivity': self.top1_eval.Sensitivity(),
                'specificity': self.top1_eval.Specificity(),
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch VGG16 Training")
    parser.add_argument('--dataset', type=str, default='ucf101flow',
                        choices=['ucf101flow', 'urfdfusion'],
                        help='dataset name (default: ucf101flow)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--temporal-weight', type=float, default=0.5,
                        metavar='TW', help='temporal network weight (default:0.5)')
    parser.add_argument('--spatial-weight', type=float, default=0.5,
                        metavar='SW', help='spatial network weight (default:0.5)')

    # optimizer params
    parser.add_argument('--temporal-lr', type=float, default=0.0001, metavar='T-LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--spatial-lr', type=float, default=0.000001, metavar='S-LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='WD',
                        help='weight decay (default: 0.0005)')
    
    # cuda, seed and logging
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'ucf101flow': 500,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        #print('Learning Rate: ', trainer.scheduler.get_last_lr())
        trainer.training(epoch)
        #trainer.scheduler.step()
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
