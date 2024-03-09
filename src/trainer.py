# open lib
import time
import torch
from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from tqdm.auto import tqdm
# third part lib
from ema_pytorch import EMA
from accelerate import Accelerator
# self lib
from model import WNet
from config import args as args_config
from metric import SSIM
from utils import Smoother, ImageLoader, TensorViser
from loss import MSELoss, SSIMLoss, GradLoss, StdLoss, SharpnessLoss, ContrastLoss
from m3fd import M3fdMix

def exists(x):
    return x is not None

class Trainer(object):
    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.accelerator = Accelerator(split_batches = self.args.if_split_batch,
                                       mixed_precision = self.args.mixed_precision)
        self.accelerator.native_amp = self.args.if_amp

        # auto encoder and vgg16(used to calculate loss)
        model_net = WNet()

        ds_train= M3fdMix()
        dl_train = DataLoader(dataset=ds_train, 
                              batch_size=self.args.train_batch, 
                              shuffle=self.args.loader_shuffle,
                              num_workers=self.args.num_workers,
                              pin_memory=True)

        # optimizer     
        opt = AdamW(model_net.parameters(), lr=self.args.lr, betas=self.args.betas)
        warm_sch = LambdaLR(opt, lr_lambda=lambda x: (x+1)/len(dl_train))
        lr_sch = MultiStepLR(opt, milestones=self.args.lr_mstone, gamma=self.args.lr_decay_gamma)

        # loss weight
        self.ssim = SSIM()
        self.ssim_fn = SSIMLoss()
        self.mse_fn = MSELoss()
        self.grad_fn = GradLoss()
        self.std_fn = StdLoss()
        self.sharp_fn = SharpnessLoss()
        self.contrast_fn = ContrastLoss()

        self.ssim_weight = 1.
        self.mse_weight = 2. 
        self.grad_weight = 0
        self.std_weight = 0
        self.sharp_weight = 0
        self.contrast_weight = 0

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            if self.args.if_ema:
                self.ema = EMA(model_net, beta=self.args.ema_decay, 
                               update_every=self.args.ema_update_every)

            self.results_folder = Path(args.save_dir)
            self.results_folder.mkdir(exist_ok = True)

        # prepare model, dataloader, optimizer with accelerator
        self.model_net = self.accelerator.prepare(model_net)
        self.opt, self.warm_sch, self.lr_sch = self.accelerator.prepare(opt, warm_sch, lr_sch)
        self.dl_train = self.accelerator.prepare(dl_train)

        # test image
        self.ir_path = 'C:/Users/wyh/Desktop/Project/UniFusion_ING/temp_data/Reek_IR.bmp'
        self.vis_path = 'C:/Users/wyh/Desktop/Project/UniFusion_ING/temp_data/Reek_VIS.bmp'

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        for self.epoch in range(1, self.args.epochs+1):
            self.model_net.train()

            current_time = time.strftime('%y%m%d@%H:%M:%S')
            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='
                  .format(self.epoch, self.args.epochs, self.args.lr, current_time, args.save_dir))
            
            len_train = len(self.dl_train)
            ssim_loss, mse_loss, grad_loss, std_loss, sharp_loss, contrast_loss, loss_sum = [Smoother(10) for _ in range(7)]
            
            with tqdm(enumerate(self.dl_train), total=len_train) as pbar:
                for batch, sample in pbar:
                    # transfer ir to vis 
                    ae_out = self.model_net(sample)

                    # get loss
                    loss = self.get_loss(sample, ae_out['ae_out'])
                    accelerator.backward(loss['loss_sum'])
                    
                    self.warm_sch.step() if self.args.if_warm_up and self.epoch == 1 else None
                    self.opt.step()
                    self.opt.zero_grad()

                    # ema update
                    if accelerator.is_main_process and self.args.if_ema:
                        self.ema.to(device)
                        self.ema.update()

                    # tqdm update
                    ssim_loss += loss['ssim_loss'].item()
                    mse_loss += loss['mse_loss'].item()
                    grad_loss += loss['grad_loss'].item()
                    std_loss += loss['std_loss'].item()
                    sharp_loss += loss['sharp_loss'].item()
                    contrast_loss += loss['contrast_loss'].item()
                    loss_sum += loss['loss_sum'].item()

                    s = f'Train | ssim: {ssim_loss:.2f} | mse: {mse_loss:.2f} | grad: {grad_loss:.2f} | std: {std_loss:.2f} | sharp: {sharp_loss:.2f} | cst: {contrast_loss:.2f} | loss: {loss_sum:.2f} | '

                    current_lr = self.opt.param_groups[0]['lr']
                    s += f'Lr: {current_lr:.2e} | ' 

                    pbar.set_description(s)

            self.lr_sch.step()
            self.save()

            if self.args.if_test: self.test()

    def save(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {'epoch': self.epoch,
                'model': self.accelerator.get_state_dict(self.model_net),
                'opt': self.opt.state_dict(),
                'ema': self.ema.state_dict() if self.args.if_ema else None,
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None}

        torch.save(data, '{}/model_{:05d}.pt'.format(self.args.save_dir, self.epoch))

    def load(self, load_epoch):
        # load model
        model = torch.load(str(self.args.load_dir + f'/model_{load_epoch:05d}.pt'))
        # load autoe and set to eval 
        self.model_net = self.accelerator.unwrap_model(self.model_net)
        self.model_net.load_state_dict(model['model'])
        self.model_net.eval()
        # load optimizer and ema
        self.opt.load_state_dict(model['opt'])
        self.ema.load_state_dict(model['ema'])
        # load scaler
        if exists(self.accelerator.scaler) and exists(model['scaler']):
            self.accelerator.scaler.load_state_dict(model['scaler'])

        print('=== pretrained model load done ===')

        # load test image
        ir_test = ImageLoader(file_path=self.ir_path, norm=False).load().to(self.accelerator.device)
        vis_test = ImageLoader(file_path=self.vis_path, norm=False).load().to(self.accelerator.device)
        # network process
        data = {'vis': vis_test, 'etc': ir_test}
        ae_out = self.model_net(data)
        # visualize
        ts_viser = TensorViser(mode='gray', save=False)
        ts_viser.visualize(ae_out['ae_out'])

        ssim_value = self.ssim(ae_out['ae_out'], ir_test) + self.ssim(ae_out['ae_out'], vis_test)
        print(f' Metric | Image SSIM: {ssim_value:.4f}')

    def get_loss(self, sample, fusion_im):
        ssim_loss = self.ssim_fn(sample['vi_y'], fusion_im) + self.ssim_fn(sample['ir_y'], fusion_im, if_dynamic=True)
        mse_loss = self.mse_fn(sample['vi_y'], fusion_im) + self.mse_fn(sample['ir_y'], fusion_im)
        grad_loss = self.grad_fn(sample['vi_y'], fusion_im) + self.grad_fn(sample['ir_y'], fusion_im)
        std_loss = self.std_fn(sample['vi_y'], fusion_im, weight=2.) + self.std_fn(sample['ir_y'], fusion_im)

        ssim_loss = ssim_loss * self.ssim_weight
        mse_loss = mse_loss * self.mse_weight
        grad_loss = grad_loss * self.grad_weight
        std_loss = std_loss * self.std_weight

        # add_loss
        # 计算各自的Sharpness Loss
        sharpness_loss_ir = self.sharp_fn(sample['ir_y'])
        sharpness_loss_visible = self.sharp_fn(sample['vi_y'])
        sharpness_loss_fused = self.sharp_fn(fusion_im)
        lambda1, lambda2 = 0.4, 0.6  # 可以调整这两个参数
        sharpness_loss = sharpness_loss_fused - lambda1 * sharpness_loss_ir - lambda2 * sharpness_loss_visible
        sharpness_loss = self.sharp_weight * sharpness_loss.abs()

        contrast_loss = torch.zeros(1, device=sharpness_loss.device)
        loss_sum = ssim_loss + mse_loss + grad_loss + std_loss + sharpness_loss + contrast_loss

        loss = {'ssim_loss': ssim_loss, 'mse_loss': mse_loss, 
                'grad_loss': grad_loss, 'std_loss': std_loss,
                'sharp_loss': sharpness_loss, 'contrast_loss': contrast_loss,
                'loss_sum': loss_sum}

        return loss


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

    trainer = Trainer(args)
    trainer.train()
    # trainer.load(load_epoch=24)




