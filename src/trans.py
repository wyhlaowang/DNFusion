import math
import torch
import torch.nn as nn
from einops import rearrange
from common import *


def pool(x, size_tensor):
    return F.adaptive_max_pool2d(x, size_tensor.shape[2:]) 


def t4d_pad(t4d, patch_h=32, patch_w=31):
    # Only Support: right-down crop
    _, _, H, W = t4d.shape
    nh = math.ceil(H / patch_h)
    nw = math.ceil(W / patch_w)
    p_down = nh * patch_h - H
    p_right = nw * patch_w - W
    padder = nn.ReflectionPad2d([0, p_right, 0, p_down])
    p_t4d = padder(t4d)
    return p_t4d


def t4d_crop(t4d, crop_h, crop_w):
    # Only Support: left-up crop
    return t4d[:, :, :crop_h, :crop_w]


def multi_group(ir,
                ir_feat,
                vis_feat,
                confidence,
                chunks=4,
                patch_list=[16,32,48,64], 
                scale=[1.,1.,1.,1.], 
                hw_ratio=[1.,1.,1.,1.], 
                eps=1e-9):
    assert chunks==len(scale)==len(hw_ratio)<=len(patch_list), '<chunks> <patch_list> <scale> <hw_ratio> must be equal'
    DEV = ir.device

    patch_w = [round(scale[ind]*ins) for ind,ins in enumerate(patch_list[:chunks])]
    patch_h = [round(hw_ratio[ind]*ins) for ind,ins in enumerate(patch_w[:chunks])]

    patch_w = [2 if ins<2 else ins for ins in patch_w]
    patch_h = [2 if ins<2 else ins for ins in patch_h]
    
    # chunk by channel
    ir_list = torch.chunk(ir_feat, chunks=chunks, dim=1)
    vis_list = torch.chunk(vis_feat, chunks=chunks, dim=1)
    conf_list = torch.chunk(confidence, chunks=chunks, dim=1)

    unifeat_list = []

    for ind, sub_ir in enumerate(ir_list):
        # get elements
        sub_vis = vis_list[ind]
        sub_conf = conf_list[ind]
        ph = patch_h[ind] if patch_h[ind]<=sub_ir.shape[2] else sub_ir.shape[2]
        pw = patch_w[ind] if patch_w[ind]<=sub_ir.shape[3] else sub_ir.shape[3]

        C = sub_ir.shape[1]

        # padder
        ir_raw = pool(ir, sub_conf).repeat(1, C, 1, 1)
        ir_pad = t4d_pad(sub_ir, ph, pw)
        ir_raw_pad = t4d_pad(ir_raw, ph, pw)
        vis_pad = t4d_pad(sub_vis, ph, pw)

        B, _, H, W = ir_pad.shape

        # converter
        nph = H // ph
        npw = W // pw
        ir_r = rearrange(ir_pad, 'b c (nph ph) (npw pw) -> b (c nph npw) ph pw', nph=nph, npw=npw, ph=ph, pw=pw)
        ir_raw_r = rearrange(ir_raw_pad, 'b c (nph ph) (npw pw) -> b (c nph npw) ph pw', nph=nph, npw=npw, ph=ph, pw=pw)
        vis_rf = rearrange(vis_pad, 'b c (nph ph) (npw pw) -> b (c nph npw) (ph pw)', nph=nph, npw=npw, ph=ph, pw=pw)
        RC = ir_r.shape[1]
        ir_rf = ir_r.reshape(B, RC, -1)
    
        # transform
        ir_std, ir_mean = [i.reshape(B, RC, 1, 1) for i in torch.std_mean(ir_rf, dim=-1, unbiased=False)]
        vis_std, vis_mean = [i.reshape(B, RC, 1, 1) for i in torch.std_mean(vis_rf, dim=-1, unbiased=False)]
        
        norm_ir = (ir_r - ir_mean) / (ir_std + eps)
        trans_ir = norm_ir * vis_std + vis_mean

        # add cos_sim
        # sim = torch.cosine_similarity(trans_ir.reshape(B, RC, -1), ir_rf, dim=-1).reshape(B, RC, 1, 1)
        com_conf = torch.where(ir_raw_r>0.7, torch.tensor([1.], device=DEV), ir_raw_r)
        com_conf = torch.where(com_conf<0.3, torch.tensor([0.], device=DEV), com_conf)
        compound_ir = (1-com_conf) * trans_ir + com_conf * ir_r
        compound_ir = trans_ir

        # rearrange
        compound_ir = rearrange(compound_ir, 'b (c nph npw) ph pw -> b c (nph ph) (npw pw)', nph=nph, npw=npw, ph=ph, pw=pw)
        compound_ir = t4d_crop(compound_ir, ir_feat.shape[2], ir_feat.shape[3])
        
        # add confidence 
        # re_conf = torch.where(ir_raw>0.7, torch.tensor([1.], device=DEV), sub_conf)
        # re_conf = torch.where(ir_raw<0.3, torch.tensor([0.], device=DEV), re_conf)
        compound_ir = sub_conf * compound_ir + (1 - sub_conf) * sub_vis
        unifeat_list.append(compound_ir)
    
    unifeat = torch.cat(unifeat_list, dim=1)

    return unifeat


class ScaleRatio(nn.Module):
    def __init__(self, a_ch, b_ch, groups=4, inner_multi=4, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.layer1 = conv_bn_relu(a_ch+b_ch, groups*inner_multi, 3, 1, 1)
        self.layer_scale = nn.Sequential(nn.AdaptiveAvgPool2d([1, 1]),
                                         nn.Flatten(),
                                         nn.Linear(groups*inner_multi, groups),
                                         nn.Tanh())
        self.layer_ratio = nn.Sequential(nn.AdaptiveAvgPool2d([1, 1]),
                                         nn.Flatten(),
                                         nn.Linear(groups*inner_multi, groups),
                                         nn.Tanh())

    def forward(self, a, b):
        f1 = self.layer1(torch.cat([a, b], dim=1))
        scale = self.layer_scale(f1)
        ratio = self.layer_ratio(f1)

        return {'scale': self.alpha*scale+1, 'ratio': self.alpha*ratio+1}


class TransInten(nn.Module):
    def __init__(self, a_dim, b_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(a_dim, a_dim), Mish())
        self.layer2 = nn.Sequential(nn.Linear(b_dim, b_dim), Mish())
        self.layer3 = nn.Sequential(nn.Linear(a_dim+b_dim, a_dim), Mish())                      
        self.layer4 = nn.Sequential(nn.Linear(a_dim, a_dim), nn.Sigmoid())                          
        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])            

    def forward(self, a, b):
        a = self.avg_pool(a).squeeze(-1).squeeze(-1)
        b = self.avg_pool(b).squeeze(-1).squeeze(-1)

        f1 = self.layer1(a)
        f2 = self.layer2(b)
        f3 = self.layer3(torch.cat([f1, f2], dim=1))
        inten = self.layer4(f3)
        
        return inten.unsqueeze(-1).unsqueeze(-1)


class TransConf(nn.Module):
    def __init__(self, a_ch, b_ch, groups=4):
        super().__init__()
        self.conf_layer = nn.Sequential(conv_bn_relu(a_ch+b_ch, a_ch, 3, 1, 1),
                                        conv_bn_relu(a_ch, a_ch, 3, 1, 1, group=True, groups=groups),
                                        nn.Conv2d(a_ch, a_ch, 3, 1, 1, groups=groups),
                                        nn.Sigmoid())

    def forward(self, vis_feat, ir_feat):
        return self.conf_layer(torch.cat([vis_feat, ir_feat], dim=1))


class Trans(nn.Module):
    def __init__(self, 
                 a_ch, 
                 b_ch, 
                 groups=4, 
                 patch_list=[8,16,32,32]):
        super().__init__()
        self.chunks = groups
        self.patch_list = patch_list
        self.intensity = TransInten(a_dim=a_ch, b_dim=b_ch)
        self.conf = TransConf(a_ch=a_ch, b_ch=b_ch, groups=groups)
        self.scale_ratio = ScaleRatio(a_ch=a_ch, b_ch=b_ch, groups=groups)

    def forward(self, vis_feat, ir_feat, ir):
        B = vis_feat.shape[0]

        inten = self.intensity(ir_feat, vis_feat)
        conf = self.conf(vis_feat, ir_feat)
        sr = self.scale_ratio(vis_feat, ir_feat)

        vis_ls = torch.chunk(vis_feat, B, dim=0)
        ir_ls = torch.chunk(ir_feat, B, dim=0)
        inten_ls = torch.chunk(inten, B, dim=0)
        conf_ls = torch.chunk(conf, B, dim=0)
        scale_ls = torch.chunk(sr['scale'], B, dim=0)
        ratio_ls = torch.chunk(sr['ratio'], B, dim=0)
        ir_raw_ls = torch.chunk(ir, B, dim=0)

        trans = []
        for i in range(B):
            trans.append(multi_group(ir_raw_ls[i],
                                     inten_ls[i]*ir_ls[i], 
                                     vis_ls[i], 
                                     conf_ls[i], 
                                     chunks=self.chunks, 
                                     patch_list=self.patch_list, 
                                     scale=scale_ls[i].flatten().tolist(), 
                                     hw_ratio=ratio_ls[i].flatten().tolist()))

        return torch.cat(trans, dim=0)

class MultiTrans(nn.Module):
    def __init__(self,
                 common_dim,
                 common_groups=4):
        super().__init__()
        self.trans1 = Trans(common_dim[0], common_dim[0], groups=common_groups)
        self.trans2 = Trans(common_dim[1], common_dim[1], groups=common_groups)
        self.trans3 = Trans(common_dim[2], common_dim[2], groups=common_groups)
        self.trans4 = Trans(common_dim[3], common_dim[3], groups=common_groups)

    def forward(self, vis_feat, ir_feat, ir):
        trans_feat = [self.trans1(vis_feat[0], ir_feat[0], ir), 
                      self.trans2(vis_feat[1], ir_feat[1], ir),
                      self.trans3(vis_feat[2], ir_feat[2], ir),
                      self.trans4(vis_feat[3], ir_feat[3], ir)]

        return trans_feat
        