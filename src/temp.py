import os
import cv2
import torch
import torch.nn as nn
from einops import rearrange


def t4d_show(t4d, batch_index=0):
    im_show = t4d[batch_index].detach().clamp(0,1).cpu().numpy()
    im_show = rearrange(im_show, 'c h w -> h w c')
    print(im_show.std())
    cv2.imshow('image', im_show)
    cv2.waitKey()


class AdaHister(nn.Module):
    def __init__(self, level=255):
        super().__init__()
        self.level = level
        self.step = torch.linspace(0, 1, level+1)
    def get_pdf_cdf(self, im):
        step_z = zip(self.step[:-1], self.step[1:])

        im = im.clamp(0, 1)
        im_f = im.flatten(1, -1)

        pdf = []
        for idx, (former, later) in enumerate(step_z):
            if idx == 0: 
                bin_num = ((im_f >= former) * (im_f <= later)).sum(dim=-1)
            else: 
                bin_num = ((im_f > former) * (im_f <= later)).sum(dim=-1)
            pdf.append(bin_num.unsqueeze(-1))

        pdf = torch.cat(pdf, dim=-1) / im_f.shape[-1]
        cdf = torch.cumsum(pdf, dim=-1)    

        return pdf, cdf

    def hist_filter(self, im, transform):
        assert self.level == transform.shape[-1], f'length is not equal ({self.level}, {transform.shape[-1]})'
        step_z = zip(self.step[:-1], self.step[1:])

        im = im.clamp(0, 1)
        im_f = torch.zeros_like(im)

        for idx, (former, later) in enumerate(step_z):
            if idx == 0: 
                mask = (im >= former) * (im <= later)
            else: 
                mask = (im > former) * (im <= later)
            im_f = im_f + mask * transform[:,idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return im_f

    def forward(self, fusion):
        _, cdf = self.get_pdf_cdf(fusion)
        filtered_fusion = self.hist_filter(fusion, cdf)

        return filtered_fusion


def main():
    print(os.getcwd())
    data_folder = './test_imgs2/TNO40'
    ir_folder = 'IR'
    vis_folder = 'VIS'
    ir_path = data_folder + '/' + ir_folder + '/'
    vis_path = data_folder + '/' + vis_folder + '/'
    file_list = os.listdir(vis_path)
    print(file_list)

    ada_hist = AdaHister()

    for i in file_list:
        vis = cv2.imread(vis_path + i, cv2.IMREAD_GRAYSCALE) / 255.
        ir = cv2.imread(ir_path + i, cv2.IMREAD_GRAYSCALE) / 255.

        B = 3
        
        vis_t = torch.from_numpy(vis).to(torch.float).to(device='cuda').repeat(B,1,1,1)
        ir_t = torch.from_numpy(ir).to(torch.float).to(device='cuda').repeat(B,1,1,1)

        cs = torch.cosine_similarity(vis_t.flatten(1,-1), ir_t.flatten(1,-1))

        hist_w = 0.4*(-torch.std(vis_t.flatten(1,-1), dim=-1, keepdim=True)).exp().unsqueeze(-1).unsqueeze(-1)
        compound = hist_w * ada_hist(vis_t) + (1 - hist_w) * vis_t

        # t4d_show(compound)

        print(' ')



if __name__ == '__main__':
    main()
