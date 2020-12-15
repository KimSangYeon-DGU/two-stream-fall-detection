class Config(object):
    @staticmethod
    def get_dataset_path(key):
        if key == 'ucf101flow':
            return '/home/aimaster/Downloads/tvl1_flow'
        elif key == 'urfdfusion':
            return '/home/aimaster/Downloads/URFD'
        else:
            print('Dataset {} not available.'.format(key))
            raise NotImplementedError

    @staticmethod
    def get_pretrained_weight_path(key):
        if key == 'ucf101flow':
            return '/home/aimaster/Desktop/DEV/vgg16-ucf101/weights/ucf101flow/vgg16/best_model/checkpoint.pth.tar'
        else:
            print('Dataset {} not available.'.format(key))
            raise NotImplementedError
