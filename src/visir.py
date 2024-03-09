import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
from torchvision.transforms import functional as TF



class VisirReader():
    def __init__(self, args):
        super(VisirReader, self).__init__()
        self.args = args

        # data reader
        raw = h5py.File(self.args.file_path, 'r')
        raw = raw['data'][:]
        raw = np.transpose(raw, (0, 3, 2, 1))

        # split test or not 
        if self.args.if_test:
            N = raw.shape[0]
            idx = np.linspace(0, N-1, N, dtype=int)
            test_idx = np.linspace(0, N-1, self.args.test_size, dtype=int)
            train_idx = np.delete(idx, test_idx)

            self.test = raw[test_idx,:,:,:]
            self.train = raw[train_idx,:,:,:]
        else:
            self.train = raw

        del raw


class VisirDataset(Dataset):
    def __init__(self, args, reader, mode='train'):
        super(VisirDataset, self).__init__()
        self.args = args
        self.reader = reader
        self.mode = mode

        if self.mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = self.args.height
        self.width = self.args.width
        self.crop_size = (self.args.crop_height, self.args.crop_width)
        self.if_augment = self.args.if_augment

    def set_mode(self, mode='train'):
        if self.mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError
        self.mode = mode

    def augment(self, vis, ir):      
        vis = Image.fromarray(vis, mode='F')
        ir = Image.fromarray(ir, mode='F')

        if self.mode == 'train':
            if self.if_augment:
                degree = np.random.uniform(-15.0, 15.0)
                vis, ir = map(lambda x: TF.rotate(x, degree, T.InterpolationMode.NEAREST), [vis, ir])

                flip = np.random.uniform(0.0, 1.0)
                vis, ir = map(lambda x: TF.hflip(x) if flip > 0.5 else x, [vis, ir])

                size = np.int(self.height * np.random.uniform(1.0, 1.5))
            else:
                size = np.int(self.height)

            tf = T.Compose([T.Resize(size),
                            T.CenterCrop(self.crop_size),
                            T.ToTensor()])

            vis, ir = map(lambda x: tf(x), [vis, ir])
        else:
            tf = T.Compose([T.CenterCrop(self.crop_size),
                            T.ToTensor()])
            vis, ir = map(lambda x: tf(x), [vis, ir])

        return vis, ir

    def __len__(self):
        if self.mode == 'train':
            length = self.reader.train.shape[0] 
        else:
            length = self.reader.test.shape[0] 

        return length

    def __getitem__(self, index):       
        vis = self.reader.train[index, :, :, 0] \
            if self.mode == 'train' else self.reader.test[index, :, :, 0]
        ir = self.reader.train[index, :, :, 1] \
            if self.mode == 'train' else self.reader.test[index, :, :, 1]

        vis, ir = self.augment(vis, ir)

        return {'vis': vis, 'etc':ir}