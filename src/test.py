import os
import cv2
import torch
from PIL import Image
from model import WNet
from torchvision import transforms as T
from config import args as args_config
from common import Hister
import torch.nn.functional as F


def load(pt_path, load_epoch, device='cuda:1'):
    model_net = WNet().to(device=device)
    model = torch.load(str(pt_path + f'/model_{load_epoch:05d}.pt'), map_location=torch.device(device))

    # load autoe and set to eval 
    model_net.load_state_dict(model['model'])
    model_net.eval()
    print('=== pretrained model load done ===')

    return model_net


def test():
    # dev = 'cpu'
    dev = 'cuda:0'

    epoch = 60

    # data_folder = 'TNO_test'
    # data_folder = 'RoadScene_test'
    # data_folder = 'irs'
    data_folder = 'rs'

    ir_folder = 'IR'
    vis_folder = 'VIS'
    pt_folder = '../experiments/final/'

    ir_path = '../test_imgs/' + data_folder + '/' + ir_folder + '/'
    vis_path = '../test_imgs/' + data_folder + '/' + vis_folder + '/'
    save_path = '../self_results/' + data_folder + '/' 

    file_list = os.listdir(ir_path)
    print(file_list)

    hs = Hister()
    model_net = load(pt_folder, epoch, dev)

    for i in file_list:
        SCALE = 1
        vis = cv2.imread(vis_path + i, cv2.IMREAD_GRAYSCALE) # only support gray image
        vis = cv2.resize(vis, (vis.shape[1]//SCALE, vis.shape[0]//SCALE))
        vis = torch.from_numpy(vis) / 255.
        vis = vis.repeat(1,1,1,1).to(device=dev)

        ir = cv2.imread(ir_path + i, cv2.IMREAD_GRAYSCALE) # only support gray image
        ir = cv2.resize(ir, (ir.shape[1]//SCALE, ir.shape[0]//SCALE))
        ir = torch.from_numpy(ir) / 255.
        ir = ir.repeat(1,1,1,1).to(device=dev)

        data = {'vi_y': vis, 'ir_y': ir}

        model_out = hs.process(model_net(data)['ae_out'])
        im = 255 * model_out.detach().cpu().squeeze(0).squeeze(0)
        im = Image.fromarray(im.numpy().astype('uint8'))
        save_file = save_path + str(epoch) + '_' + i
        im.save(save_file, quality=100)
        print(f'Info: {save_file}')


if __name__ == '__main__':
    # config
    args = args_config

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':',  getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    test()

