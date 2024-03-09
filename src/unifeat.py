import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from common import conv_bn_relu


def t4d_pad(t4d, patch_h=32, patch_w=31):
    # Only Support: right-down crop
    _, _, H, W = t4d.shape

    nh = math.ceil(H / patch_h)
    nw = math.ceil(W / patch_w)
    p_down = nh * patch_h - H
    p_right = nw * patch_w - W
    p_t4d = F.pad(t4d, pad=[0, p_right, 0, p_down], mode='reflect')

    return p_t4d


def t4d_crop(t4d, crop_h, crop_w):
    # Only Support: left-up crop
    return t4d[:, :, :crop_h, :crop_w]


class UniT(nn.Module):
    def __init__(self, in_dim, patch_size):
        super().__init__()

    def forward(self):
        pass


class HWScaleRatio(nn.Module):
    def __init__(self, in_dim, groups=4, expand=4, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.layer = nn.Sequential(conv_bn_relu(in_dim*2, groups*expand, 3, 2, 1),
                                   conv_bn_relu(groups*expand, groups*expand, 3, 1, 1))
        self.layer_scale = nn.Sequential(nn.AdaptiveAvgPool2d([1, 1]),
                                         nn.Flatten(1,-1),
                                         nn.Linear(groups*expand, groups*expand),
                                         nn.GELU(),
                                         nn.Linear(groups*expand, groups),
                                         nn.Tanh())
        self.layer_ratio = nn.Sequential(nn.AdaptiveAvgPool2d([1, 1]),
                                         nn.Flatten(1,-1),
                                         nn.Linear(groups*expand, groups*expand),
                                         nn.GELU(),
                                         nn.Linear(groups*expand, groups),                                         
                                         nn.Tanh())
        
    def forward(self, feat1, feat2):
        feat_cat = self.layer(torch.cat([feat1, feat2], dim=1))
        scale_base = self.layer_scale(feat_cat)
        ratio_base = self.layer_ratio(feat_cat)

        scale = self.alpha * scale_base + 1
        ratio = self.alpha * ratio_base + 1

        return scale, ratio


class DeformCrossAttention(nn.Module):
    def __init__(self, 
                 in_dim, 
                 dim, 
                 groups=1,
                 head=4):
        super().__init__()
        # <groups> must be equal to <1>
        self.head = head
        self.get_scale_ratio = HWScaleRatio(in_dim=in_dim, 
                                            groups=groups, 
                                            expand=4, 
                                            alpha=0.9)

        self.vis_to_qk = nn.Sequential(conv_bn_relu(in_dim, dim, 5, 1, 2),
                                       conv_bn_relu(dim, dim*2, 5, 1, 2),
                                       conv_bn_relu(dim*2, dim*2, 5, 1, 2, bn=False, relu=True),
                                       conv_bn_relu(dim*2, dim*2, 5, 1, 2, bn=False, relu=False))
        
        self.ir_to_qk = nn.Sequential(conv_bn_relu(in_dim, dim*2, 5, 1, 2),
                                      conv_bn_relu(dim*2, dim*2, 5, 1, 2),
                                      conv_bn_relu(dim*2, dim*2, 5, 1, 2, bn=False, relu=True),
                                      conv_bn_relu(dim*2, dim*2, 5, 1, 2, bn=False, relu=False))       

        self.to_global_v = nn.Sequential(conv_bn_relu(in_dim*2, dim*4, 5, 1, 2),
                                      conv_bn_relu(dim*4, dim*4, 5, 1, 2),
                                      conv_bn_relu(dim*4, dim*4, 5, 1, 2, bn=False, relu=True),
                                      conv_bn_relu(dim*4, dim*4, 5, 1, 2, bn=False, relu=False))

        self.softmax = nn.Softmax(dim=-1)

    def attention(self,
                  vis_feat,
                  ir_feat,
                  patch=16, 
                  min_patch=8,
                  hw_scale=1., 
                  hw_ratio=1., 
                  head=4):
        # pad feat
        patch_w = max(round(hw_scale*patch), min_patch)
        patch_h = max(round(hw_ratio*patch_w), min_patch)

        vis_feat_pad = t4d_pad(vis_feat, patch_h, patch_w)
        ir_feat_pad = t4d_pad(ir_feat, patch_h, patch_w)        
        
        n_patch_w = vis_feat_pad.shape[-1] // patch_w 
        n_patch_h = vis_feat_pad.shape[-2] // patch_h

        # to qk and token
        vis_qk = self.vis_to_qk(vis_feat_pad)
        ir_qk = self.ir_to_qk(ir_feat_pad)

        vis_qk_tuple = torch.chunk(vis_qk, 2, dim=1)
        ir_qk_tuple = torch.chunk(ir_qk, 2, dim=1)

        head_dim = (vis_qk_tuple[0].shape[1] // head) * patch_h * patch_w
        scale = head_dim ** -0.5

        # rearrange func
        qk_to_token = nn.Sequential(Rearrange('b (h d) (nph ph) (npw pw) -> b h (nph npw) (d ph pw)', 
                                              h=head, ph=patch_h, pw=patch_w, nph=n_patch_h, npw=n_patch_w),
                                    nn.LayerNorm(head_dim))
        v_to_token = nn.Sequential(Rearrange('b (h n d) (nph ph) (npw pw) -> b h (n nph npw) (d ph pw)', 
                                             h=head, n=2, ph=patch_h, pw=patch_w, nph=n_patch_h, npw=n_patch_w),
                                    nn.LayerNorm(head_dim))
        to_tensor = nn.Sequential(Rearrange('b h (nph npw) (d ph pw) -> b (h d) (nph ph) (npw pw)', 
                                            h=head, ph=patch_h, pw=patch_w, nph=n_patch_h, npw=n_patch_w))
        
        vis_q, vis_k = [qk_to_token(ins) for ins in vis_qk_tuple]
        ir_q, ir_k = [qk_to_token(ins) for ins in ir_qk_tuple]

        # to global v
        global_v = self.to_global_v(torch.cat([vis_feat_pad, ir_feat_pad], dim=1))
        global_v_tuple = torch.chunk(global_v, 2, dim=1)
        vis_gv, ir_gv = [v_to_token(ins) for ins in global_v_tuple]

        # v-v, v-r, r-r, r-v similarity
        vv_dots = torch.matmul(vis_q, vis_k.transpose(-1,-2)) * scale
        vr_dots = torch.matmul(vis_q, ir_k.transpose(-1,-2)) * scale

        rr_dots = torch.matmul(ir_q, ir_k.transpose(-1,-2)) * scale
        rv_dots = torch.matmul(ir_q, vis_k.transpose(-1,-2)) * scale

        vcross_atten = self.softmax(torch.cat([vv_dots, vr_dots], dim=-1))
        rcross_atten = self.softmax(torch.cat([rr_dots, rv_dots], dim=-1))

        vis_out = to_tensor(torch.matmul(vcross_atten, vis_gv))
        ir_out = to_tensor(torch.matmul(rcross_atten, ir_gv))

        unifeat = torch.cat([vis_out, ir_out], dim=1)        
        return unifeat

    def forward(self, vis_feat, ir_feat):
        H, W = vis_feat.shape[-2:]
        hw_scale, hw_ratio = self.get_scale_ratio(vis_feat, ir_feat)

        unifeat_list = []
        for i in range(hw_scale.shape[0]):
            print(i)
            unifeat_list.append(self.attention(vis_feat[i].unsqueeze(0), 
                                               ir_feat[i].unsqueeze(0), 
                                               hw_scale=hw_scale[i].item(), 
                                               hw_ratio=hw_ratio[i].item(), 
                                               head=self.head))

        unifeat = torch.cat(unifeat_list, dim=0)
        return t4d_crop(unifeat, H, W)


    

if __name__ == '__main__':
    vis_feat = torch.rand([2,8,43,43])
    ir_feat = torch.rand([2,8,43,43])

    attention = DeformCrossAttention(in_dim=8, 
                                     dim=32, 
                                     groups=1,
                                     head=4)
    
    unifeat = attention(vis_feat, ir_feat)

    print(unifeat.shape)

