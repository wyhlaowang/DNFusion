import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
from torchvision.transforms import functional as TF
from einops import rearrange


class Smoother():
    # smooth data
    def __init__(self, windowsize=200):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0
    
    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self
    
    def __float__(self):
        return float(self.data.mean())
    
    def __format__(self, f):
        return self.__float__().__format__(f)


class ImageLoader():
    def __init__(self, file_path, size=None, crop=None, norm=False, tri_ch=False):
        assert file_path is not None, 'file_path is not allowed to be empty! '
        self.file_path = file_path
        self.size = size 
        self.crop = crop
        self.norm = norm
        self.tri_ch = tri_ch

    def load(self):      
        image = Image.open(self.file_path)
        C, H, W = len(image.getbands()), image.height, image.width

        if self.norm and C == 1:
            tn = T.Normalize(0.449, 0.226)
        elif self.norm and C == 3:
            tn = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
            tn = T.Normalize(0, 1)

        tf = T.Compose([T.Resize(self.size) if self.size is not None else T.Resize((H, W)),
                        T.CenterCrop(self.crop) if self.crop is not None else T.CenterCrop((H, W)),
                        T.ToTensor(),
                        tn])
        
        image = tf(image)
        image = image.unsqueeze(0)  
        return image.repeat(1, 3, 1, 1) if self.tri_ch and C == 1 else image  


class TensorViser():
    def __init__(self, mode='gray', save=True, save_path=None, show=True):
        assert mode in ['gray', 'rgb'], 'only "gray", "rgb" mode are supported'
        self.mode = mode
        self.save = save
        self.save_path = save_path if save_path is not None else '.'
        self.show = show

    def visualize(self, show_t):    
        B, C, _, _ = show_t.shape
        assert B == 1, 'only single batch tensor is supported'
        im = show_t.detach().cpu().squeeze(0)
        im = rearrange(255*im, 'c h w -> h w c')
        im = torch.mean(im, dim=-1) if self.mode == 'gray' else im
        im = Image.fromarray(im.numpy().astype('uint8'))
        current_time = time.strftime('%m%d_%H%M')
        save_file = self.save_path + '/' + current_time + '_trail.png'
        im.save(save_file) if self.save else None
        im.show() if self.show else None
