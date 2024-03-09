import torch
import torch.nn as nn
import torch.nn.functional as F
from trans import MultiTrans
from common import *
from canny import CannyDetector
from modulated_deform_conv_func import ModulatedDeformConvFunction
from einops import rearrange


def pool(x, size_tensor):
    return F.adaptive_max_pool2d(x, size_tensor.shape[2:]) 


class CAM(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=0.5, if_sigmoid=True):
        super().__init__()
        inner_dim = int(in_dim*expansion)
        self.sam = nn.Sequential(nn.AdaptiveAvgPool2d([1,1]),
                                 nn.Flatten(1,-1),
                                 nn.Linear(in_dim, inner_dim),
                                 nn.CELU(),
                                 nn.Linear(inner_dim, inner_dim),
                                 nn.CELU(),
                                 nn.Linear(inner_dim, out_dim),
                                 nn.Sigmoid() if if_sigmoid else nn.Sequential())
        
    def forward(self, x):
        return self.sam(x).unsqueeze(-1).unsqueeze(-1)


class AttentionMap(nn.Module):
    def __init__(self, in_dim=1, dim=[2,4,8]):
        super().__init__()
        self.mapping = nn.Sequential(conv_bn_relu(in_dim, dim[0], 3, 2, 1),
                                     conv_bn_relu(dim[0], dim[1], 3, 2, 1),
                                     conv_bn_relu(dim[1], dim[2], 3, 2, 1, bn=False, relu=True))

    def forward(self, vis, ir):
        vis_feat = self.mapping(vis)
        ir_feat = self.mapping(ir)

        # sim = F.pairwise_distance(vis_feat, ir_feat, keepdim=True)
        # priat_map = F.interpolate(sim, vis.shape[2:], mode='bilinear', align_corners=False)
        # coat_map = 1 - priat_map

        sim = F.cosine_similarity(vis_feat, ir_feat, dim=1).unsqueeze(1)
        coat_map = F.interpolate(sim, vis.shape[2:], mode='bilinear', align_corners=False)
        priat_map = 1 - coat_map
        return priat_map+1, coat_map+1


class DenseBolck(nn.Module):
    def __init__(self, in_dim=1, layer_dim=[8,16,32,64], squeeze=2, group=False, groups=4, expansion=2):
        super().__init__()
        squeeze_dim = [i//squeeze if i>=squeeze else 1 for i in layer_dim]

        self.layer1 = conv_bn_relu(in_dim, layer_dim[0], 3, 1, 1, bn=False)
        self.layer2 = conv_bn_relu(layer_dim[0]+squeeze_dim[0], layer_dim[1], 3, 1, 1, group=group, groups=groups)
        self.layer3 = conv_bn_relu(layer_dim[1]+squeeze_dim[0]+squeeze_dim[1], layer_dim[2], 3, 2, 1, group=group, groups=groups)
        self.layer4 = conv_bn_relu(layer_dim[2]+squeeze_dim[0]+squeeze_dim[1]+squeeze_dim[2], layer_dim[3], 3, 2, 1, group=group, groups=groups)

        self.l1_squeeze = conv_bn_relu(layer_dim[0], squeeze_dim[0], 3, 1, 1)
        self.l2_squeeze = conv_bn_relu(layer_dim[1], squeeze_dim[1], 3, 1, 1)
        self.l3_squeeze = conv_bn_relu(layer_dim[2], squeeze_dim[2], 3, 1, 1)

        self.l1s_attn = CAM(squeeze_dim[0], squeeze_dim[0]*3, expansion=1.5)
        self.l2s_attn = CAM(squeeze_dim[1], squeeze_dim[1]*2, expansion=1)
        self.l3s_attn = CAM(squeeze_dim[2], squeeze_dim[2], expansion=0.5)

    def forward(self, x):
        f1 = self.layer1(x)
        f1_s = self.l1_squeeze(f1)
        f1_attn = self.l1s_attn(f1_s)
        f1_attn = torch.chunk(f1_attn, 3, 1)

        f2 = self.layer2(torch.cat([f1, 0*f1_s*f1_attn[0]], dim=1))
        f2_s = self.l2_squeeze(f2)
        f2_attn = self.l2s_attn(f2_s)
        f2_attn = torch.chunk(f2_attn, 2, 1)

        f3 = self.layer3(torch.cat([f2, 0*pool(f1_s*f1_attn[1], f2), 0*f2_s*f2_attn[0]], dim=1))
        f3_s = self.l3_squeeze(f3)
        f3_attn = self.l3s_attn(f3_s)

        f4 = self.layer4(torch.cat([f3, 0*pool(f1_s*f1_attn[2], f3), 0*pool(f2_s*f2_attn[1], f3), 0*f3_s*f3_attn], dim=1))

        return [f1, f2, f3, f4]


class SEdgeDetector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.squeeze = nn.Sequential(conv_bn_relu(in_dim, 1, 3, 1, 1),
                                     conv_bn_relu(1, 1, 3, 1, 1),
                                     conv_bn_relu(1, 1, 3, 1, 1, bn=False))
    
    def forward(self, x, edge_op):
        x_sq = self.squeeze(x)
        x_edge = edge_op(x_sq)
        return x_edge


class FeatureExtractor(nn.Module):
    def __init__(self, 
                 vis_dim=[8,16,24,32], 
                 ir_dim=[8,16,24,32], 
                 common_dim=[8,16,32,64],
                 common_group=True,
                 common_groups=4):
        super().__init__()
        self.vis_feat = DenseBolck(in_dim=1, layer_dim=vis_dim)
        self.ir_feat = DenseBolck(in_dim=1, layer_dim=ir_dim)
        self.common_feat = DenseBolck(in_dim=1, layer_dim=common_dim, group=common_group, groups=common_groups)

    def forward(self, vis_private, ir_private, vis_common, ir_common):
        vis_prifeat = self.vis_feat(vis_private)
        ir_prifeat = self.ir_feat(ir_private)

        vis_cofeat = self.common_feat(vis_common)
        ir_cofeat = self.common_feat(ir_common)

        return {'vis_prifeat': vis_prifeat, 'ir_prifeat': ir_prifeat, 
                'vis_cofeat': vis_cofeat, 'ir_cofeat': ir_cofeat}


class EdgeConv(nn.Module):
    def __init__(self, in_dim=2, inner_dim=4, out_dim=4, if_visir_edge=True):
        super().__init__()
        self.if_visir_edge = if_visir_edge
        self.edge_detect = SEdgeDetector(in_dim)
        self.edge_conv = conv_bn_relu(in_dim, inner_dim, 3, 1, 1)
        self.vr_conv = conv_bn_relu(in_dim, inner_dim, 3, 1, 1) if if_visir_edge else nn.Sequential()
        self.feat = conv_bn_relu(inner_dim*2 if if_visir_edge else inner_dim, out_dim, 3, 1, 1)

    def forward(self, x, edge_op, visir_edge=None):
        if self.if_visir_edge == True:
            assert visir_edge != None, 'need <visir_edge>'
            assert visir_edge.shape[1] == 1, '<visir_edge channle> need equal to 1'

        x_max, _ = x.flatten(1,-1).max(dim=-1)
        x_max = x_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x_edge = (x_max * self.edge_detect(x, edge_op)) + x
        x_edge_feat = self.edge_conv(x_edge)

        if self.if_visir_edge:
            vr_egde = (x_max * visir_edge) + x
            vr_edge_feat = self.vr_conv(vr_egde)
            return self.feat(torch.cat([x_edge_feat, vr_edge_feat], dim=1))
        else:
            return self.feat(x_edge_feat)


class Reconstructor(nn.Module):
    def __init__(self, 
                 vis_dim=[8,16,24,32], 
                 ir_dim=[8,16,24,32], 
                 common_dim=[8,16,32,64], 
                 inner_dim1=[24,32,32,64], 
                 inner_dim2=[24,32,32,64], 
                 out_dim=[8,4,2,1]):
        super().__init__()
        vif_dim = [ins+vis_dim[ind] for ind, ins in enumerate(common_dim)]
        irf_dim = [ins+ir_dim[ind] for ind, ins in enumerate(common_dim)]
        feat_dim = [x+y for x, y in zip(vis_dim, ir_dim)]

        self.covis_cam_list = nn.ModuleList(CAM(cdim, cdim, expansion=0.5) for cdim in common_dim)
        self.comir_cam_list = nn.ModuleList(CAM(cdim, cdim, expansion=0.5) for cdim in common_dim)

        # vis and fusion
        self.vis_fusion4 = conv_bn_relu(vif_dim[3], vis_dim[3], 3, 1, 1)
        self.ir_fusion4 = conv_bn_relu(irf_dim[3], ir_dim[3], 3, 1, 1)
        # 1/4, feat_dim -> 1/2, inner_dim
        self.dec4_1 = convt_bn_relu(feat_dim[3], inner_dim1[3], 3, 2, 1, output_padding=1)
        # 1/2, inner_dim -> 1, inner_dim
        self.dec4_2 = convt_bn_relu(inner_dim1[3], inner_dim2[3], 3, 2, 1, output_padding=1)

        # vis and fusion
        self.vis_fusion3 = conv_bn_relu(vif_dim[2], vis_dim[2], 3, 1, 1)
        self.ir_fusion3 = conv_bn_relu(irf_dim[2], ir_dim[2], 3, 1, 1)
        # 1/2, feat_dim -> 1, inner_dim
        self.dec3_1 = convt_bn_relu(feat_dim[2]+inner_dim1[3], inner_dim1[2], 3, 2, 1, output_padding=1)
        # 1, inner_dim -> 1, inner_dim
        self.dec3_2 = conv_bn_relu(inner_dim1[2]+inner_dim2[3], inner_dim2[2], 3, 1, 1)

        # vis and fusion
        self.vis_fusion2 = conv_bn_relu(vif_dim[1], vis_dim[1], 3, 1, 1)
        self.ir_fusion2 = conv_bn_relu(irf_dim[1], ir_dim[1], 3, 1, 1)
        # 1, feat_dim -> 1, inner_dim
        self.dec2_1 = convt_bn_relu(feat_dim[1]+inner_dim1[2], inner_dim1[1], 3, 1, 1)
        # 1, inner_dim -> 1, inner_dim
        self.dec2_2 = conv_bn_relu(inner_dim1[1]+inner_dim2[2], inner_dim2[1], 3, 1, 1)
        
        # vis and fusion
        self.vis_fusion1 = conv_bn_relu(vif_dim[0], vis_dim[0], 3, 1, 1)
        self.ir_fusion1 = conv_bn_relu(irf_dim[0], ir_dim[0], 3, 1, 1)        
        # 1, feat_dim -> 1, inner_dim
        self.dec1_1 = convt_bn_relu(feat_dim[0]+inner_dim1[1], inner_dim1[0], 3, 1, 1)
        # 1, inner_dim -> 1, inner_dim
        self.dec1_2 = conv_bn_relu(inner_dim1[0]+inner_dim2[1], inner_dim2[0], 3, 1, 1)


    def _crop(self, t4d, obj_h, obj_w):
        # Only Support: center crop
        H, W = t4d.shape[2:]
        ch = (H-obj_h) // 2
        cw = (W-obj_w) // 2
        t4d = t4d[:, :, ch:ch+obj_h, :]
        t4d = t4d[:, :, :, cw:cw+obj_w]
        return t4d

    def forward(self, vis_feat, ir_feat, common_feat, edge_op, visir_edge=None):
        assert len(vis_feat) == len(ir_feat) == len(common_feat) == 4

        H0, W0 = common_feat[0].shape[2:]
        H1, W1 = common_feat[1].shape[2:]
        H2, W2 = common_feat[2].shape[2:]
        H3, W3 = common_feat[3].shape[2:]

        vic_feat = []
        for ind, ins in enumerate(common_feat):
            vic_feat.append(torch.cat([ins*self.covis_cam_list[ind](ins)*0, vis_feat[ind]], dim=1))

        irc_feat = []
        for ind, ins in enumerate(common_feat):
            irc_feat.append(torch.cat([ins*self.comir_cam_list[ind](ins)*0, ir_feat[ind]], dim=1))

        # decoder3
        vif4 = self.vis_fusion4(vic_feat[3])
        irf4 = self.ir_fusion4(irc_feat[3])
        fd4_1 = self._crop(self.dec4_1(torch.cat([vif4, irf4], dim=1)), H2, W2)
        fd4_2 = self._crop(self.dec4_2(fd4_1), H1, W1)

        # decoder2
        vif3 = self.vis_fusion3(vic_feat[2])
        irf3 = self.ir_fusion3(irc_feat[2])
        fd3_1 = self._crop(self.dec3_1(torch.cat([vif3, irf3, fd4_1], dim=1)), H1, W1)
        fd3_2 = self.dec3_2(torch.cat([fd3_1, fd4_2], dim=1))

        # decoder1
        vif2 = self.vis_fusion2(vic_feat[1])
        irf2 = self.ir_fusion2(irc_feat[1])
        fd2_1 = self.dec2_1(torch.cat([vif2, irf2, fd3_1], dim=1))
        fd2_2 = self.dec2_2(torch.cat([fd2_1, fd3_2], dim=1))

        # decoder0
        vif1 = self.vis_fusion1(vic_feat[0])
        irf1 = self.ir_fusion1(irc_feat[0])
        fd1_1 = self.dec1_1(torch.cat([vif1, irf1, fd2_1], dim=1))
        fd1_2 = self.dec1_2(torch.cat([fd1_1, fd2_2], dim=1))

        feat = torch.cat([fd4_2, fd3_2, fd2_2, fd1_2], dim=1)

        return feat


# class Reconstructor(nn.Module):
#     def __init__(self, 
#                  vis_dim=[8,16,24,32], 
#                  ir_dim=[8,16,24,32], 
#                  common_dim=[8,16,32,64], 
#                  inner_dim1=[24,32,32,64], 
#                  inner_dim2=[24,32,32,64], 
#                  out_dim=[8,4,2,1]):
#         super().__init__()
#         vif_dim = [ins+vis_dim[ind] for ind, ins in enumerate(common_dim)]
#         irf_dim = [ins+ir_dim[ind] for ind, ins in enumerate(common_dim)]
#         feat_dim = [x+y for x, y in zip(vis_dim, ir_dim)]

#         self.covis_cam_list = nn.ModuleList(CAM(cdim, cdim, expansion=0.5) for cdim in common_dim)
#         self.comir_cam_list = nn.ModuleList(CAM(cdim, cdim, expansion=0.5) for cdim in common_dim)

#         # vis and fusion
#         self.vis_fusion4 = conv_bn_relu(vif_dim[3], vis_dim[3], 3, 1, 1)
#         self.ir_fusion4 = conv_bn_relu(irf_dim[3], ir_dim[3], 3, 1, 1)
#         # 1/4, feat_dim -> 1/2, inner_dim
#         self.dec4_1 = convt_bn_relu(feat_dim[3], inner_dim1[3], 3, 2, 1, output_padding=1)
#         # 1/2, inner_dim -> 1, inner_dim
#         self.dec4_2 = convt_bn_relu(inner_dim1[3], inner_dim2[3], 3, 2, 1, output_padding=1)
#         # 1, inner_dim -> 1, out_dim
#         self.dec4_3 = EdgeConv(inner_dim2[3], out_dim[3], out_dim[3], if_visir_edge=False)

#         # vis and fusion
#         self.vis_fusion3 = conv_bn_relu(vif_dim[2], vis_dim[2], 3, 1, 1)
#         self.ir_fusion3 = conv_bn_relu(irf_dim[2], ir_dim[2], 3, 1, 1)
#         # 1/2, feat_dim -> 1, inner_dim
#         self.dec3_1 = convt_bn_relu(feat_dim[2]+inner_dim1[3], inner_dim1[2], 3, 2, 1, output_padding=1)
#         # 1, inner_dim -> 1, inner_dim
#         self.dec3_2 = conv_bn_relu(inner_dim1[2]+inner_dim2[3], inner_dim2[2], 3, 1, 1)
#         # 1, inner_dim -> 1, out_dim
#         self.dec3_3 = EdgeConv(inner_dim2[2], out_dim[2], out_dim[2], if_visir_edge=False)

#         # vis and fusion
#         self.vis_fusion2 = conv_bn_relu(vif_dim[1], vis_dim[1], 3, 1, 1)
#         self.ir_fusion2 = conv_bn_relu(irf_dim[1], ir_dim[1], 3, 1, 1)
#         # 1, feat_dim -> 1, inner_dim
#         self.dec2_1 = convt_bn_relu(feat_dim[1]+inner_dim1[2], inner_dim1[1], 3, 1, 1)
#         # 1, inner_dim -> 1, inner_dim
#         self.dec2_2 = conv_bn_relu(inner_dim1[1]+inner_dim2[2], inner_dim2[1], 3, 1, 1)
#         # 1, inner_dim -> 1, out_dim
#         self.dec2_3 = EdgeConv(inner_dim2[1], out_dim[1], out_dim[1], if_visir_edge=False)
        
#         # vis and fusion
#         self.vis_fusion1 = conv_bn_relu(vif_dim[0], vis_dim[0], 3, 1, 1)
#         self.ir_fusion1 = conv_bn_relu(irf_dim[0], ir_dim[0], 3, 1, 1)        
#         # 1, feat_dim -> 1, inner_dim
#         self.dec1_1 = convt_bn_relu(feat_dim[0]+inner_dim1[1], inner_dim1[0], 3, 1, 1)
#         # 1, inner_dim -> 1, inner_dim
#         self.dec1_2 = conv_bn_relu(inner_dim1[0]+inner_dim2[1], inner_dim2[0], 3, 1, 1)
#         # 1, inner_dim -> 1, out_dim
#         self.dec1_3 = EdgeConv(inner_dim2[0], out_dim[0], out_dim[0], if_visir_edge=True)


#     def _crop(self, t4d, obj_h, obj_w):
#         # Only Support: center crop
#         H, W = t4d.shape[2:]
#         ch = (H-obj_h) // 2
#         cw = (W-obj_w) // 2
#         t4d = t4d[:, :, ch:ch+obj_h, :]
#         t4d = t4d[:, :, :, cw:cw+obj_w]
#         return t4d

#     def forward(self, vis_feat, ir_feat, common_feat, edge_op, visir_edge=None):
#         assert len(vis_feat) == len(ir_feat) == len(common_feat) == 4

#         H0, W0 = common_feat[0].shape[2:]
#         H1, W1 = common_feat[1].shape[2:]
#         H2, W2 = common_feat[2].shape[2:]
#         H3, W3 = common_feat[3].shape[2:]

#         vic_feat = []
#         for ind, ins in enumerate(common_feat):
#             vic_feat.append(torch.cat([ins*self.covis_cam_list[ind](ins), vis_feat[ind]], dim=1))

#         irc_feat = []
#         for ind, ins in enumerate(common_feat):
#             irc_feat.append(torch.cat([ins*self.comir_cam_list[ind](ins), ir_feat[ind]], dim=1))

#         # decoder3
#         vif4 = self.vis_fusion4(vic_feat[3])
#         irf4 = self.ir_fusion4(irc_feat[3])
#         fd4_1 = self._crop(self.dec4_1(torch.cat([vif4, irf4], dim=1)), H2, W2)
#         fd4_2 = self._crop(self.dec4_2(fd4_1), H1, W1)
#         fd4_3 = self.dec4_3(fd4_2, edge_op)

#         # decoder2
#         vif3 = self.vis_fusion3(vic_feat[2])
#         irf3 = self.ir_fusion3(irc_feat[2])
#         fd3_1 = self._crop(self.dec3_1(torch.cat([vif3, irf3, fd4_1], dim=1)), H1, W1)
#         fd3_2 = self.dec3_2(torch.cat([fd3_1, fd4_2], dim=1))
#         fd3_3 = self.dec3_3(fd3_2, edge_op)

#         # decoder1
#         vif2 = self.vis_fusion2(vic_feat[1])
#         irf2 = self.ir_fusion2(irc_feat[1])
#         fd2_1 = self.dec2_1(torch.cat([vif2, irf2, fd3_1], dim=1))
#         fd2_2 = self.dec2_2(torch.cat([fd2_1, fd3_2], dim=1))
#         fd2_3 = self.dec2_3(fd2_2, edge_op)

#         # decoder0
#         vif1 = self.vis_fusion1(vic_feat[0])
#         irf1 = self.ir_fusion1(irc_feat[0])
#         fd1_1 = self.dec1_1(torch.cat([vif1, irf1, fd2_1], dim=1))
#         fd1_2 = self.dec1_2(torch.cat([fd1_1, fd2_2], dim=1))
#         fd1_3 = self.dec1_3(fd1_2, edge_op, visir_edge)

#         feat = torch.cat([fd4_3, fd3_3, fd2_3, fd1_3], dim=1)

#         return feat


class Fusion(nn.Module):
    def __init__(self, in_dim=32, ne_num=8, aff_gamma=0.5):
        super().__init__()
        self.ne_num = ne_num
        self.idx_ref = ne_num // 2
        self.aff_scale_const = nn.Parameter(aff_gamma*ne_num*torch.ones(1))

        self.gen_sigma = nn.Sequential(CAM(in_dim, 1, 0.5, if_sigmoid=False),
                                       nn.ReLU())
        self.fuse = nn.Sequential(conv_bn_relu(in_dim, in_dim, 3, 1, 1),
                                  conv_bn_relu(in_dim, 1, 3, 1, 1, bn=False, relu=False),
                                  nn.Tanh())           
        self.guid = nn.Sequential(conv_bn_relu(in_dim, in_dim, 5, 1, 2),
                                  conv_bn_relu(in_dim, ne_num, 5, 1, 2),
                                  conv_bn_relu(ne_num, ne_num, 5, 1, 2, bn=False, relu=False))         
        self.conv_off_aff = nn.Sequential(conv_bn_relu(ne_num, ne_num, 5, 1, 2),
                                          conv_bn_relu(ne_num, ne_num*3, 5, 1, 2),
                                          conv_bn_relu(ne_num*3, ne_num*3, 5, 1, 2, bn=False, relu=False))         
        self.conv_aff = nn.Conv2d(ne_num, ne_num, 3, 1, 1)

        self.w = nn.Parameter(torch.ones((1, 1, 3, 3)))
        self.b = nn.Parameter(torch.zeros(1))

        self.w.requires_grad = False
        self.b.requires_grad = False

    def create_gauss(self, window_size, sigma=1, dev='cuda'):
        item_num = window_size**2

        bias = torch.tensor([-((x-window_size//2)**2) for x in range(window_size)], device=dev).unsqueeze(0)
        gauss = (bias / (2*sigma**2)).exp()

        win_1d = (gauss/gauss.sum(dim=1, keepdim=True)).unsqueeze(-1)
        win_2d = win_1d.bmm(win_1d.transpose(-2,-1)).flatten(1, -1)
        win_2d = list(torch.chunk(win_2d, item_num, 1))
        win_2d.pop(item_num//2)
        win_2d = torch.cat(win_2d, dim=1)

        return win_2d

    def get_off_aff(self, guid, sigma, sigma_gamma=3, window_size=3):
        B, _, H, W = guid.shape
        DEV = guid.device

        off_aff = self.conv_off_aff(guid)
        o1, o2, aff = torch.chunk(off_aff, 3, 1)

        # Add zero reference offset
        offset = torch.cat((o1, o2), dim=1).view(B, self.ne_num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.ne_num, dim=1))
        list_offset.insert(self.idx_ref, torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        # get gauss kernel
        sigma = sigma + sigma_gamma
        gau_ker = self.create_gauss(window_size, sigma, DEV)
        gau_ker = gau_ker.unsqueeze(-1).unsqueeze(-1)

        # get aff
        aff = torch.tanh(aff) / (self.aff_scale_const + 1e-5)
        aff = aff + gau_ker
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-5
        aff_abs_sum[aff_abs_sum < 1.0] = 1.0
        aff = aff / aff_abs_sum
        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum
        list_aff = list(torch.chunk(aff, self.ne_num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def gauss_filter(self, fusion, offset, aff, times=1):
        for _ in range(times):
            fusion = ModulatedDeformConvFunction.apply(fusion, 
                                                       offset, 
                                                       aff, 
                                                       self.w, self.b, 1, 1, 1, 1, 1, 64)     

        return fusion
    
    def get_aff(self, guid, sigma, window_size=3):
        B, _, H, W = guid.shape
        DEV = guid.device

        aff = self.conv_aff(guid)

        # get gauss kernel
        gau_ker = self.create_gauss(window_size, sigma, DEV)
        gau_ker = gau_ker.unsqueeze(-1).unsqueeze(-1)

        # get aff
        aff = torch.tanh(aff) / (self.aff_scale_const + 1e-5)
        aff = aff + gau_ker
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-5
        aff_abs_sum[aff_abs_sum < 1.0] = 1.0
        aff = aff / aff_abs_sum
        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum
        list_aff = list(torch.chunk(aff, self.ne_num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return aff
    
    def dynamic_filter(self, image, weights):
        """
        对图像应用动态滤波。
        image: 输入图像，形状为 [B, 1, H, W]
        weights: 权重矩阵，形状为 [B, 9, H, W]
        """
        B, C, H, W = image.shape

        # 展开图像
        unfolded_image = F.unfold(image, kernel_size=3, padding=1, stride=1)

        # 调整权重以匹配展开的图像
        weights = rearrange(weights, "b c h w -> b c (h w)")

        filtered_image = rearrange(torch.sum(unfolded_image * weights, dim=1, keepdim=True), "b c (h w) -> b c h w", h=H, w=W)

        return filtered_image

    def forward(self, feat):
        guid = self.guid(feat)
        fusion = self.fuse(feat)
        sigma = self.gen_sigma(feat.clone().detach()).squeeze(-1).squeeze(-1) + 1e-3

        offset, aff = self.get_off_aff(guid, sigma=sigma, sigma_gamma=0.2, window_size=3)
        gfusion = self.gauss_filter(fusion, offset, aff).abs()

        # aff = self.get_aff(guid, sigma=0.2, window_size=3)
        # gfusion = self.dynamic_filter(fusion, aff).abs()

        # return gfusion
        return fusion


class WNet(nn.Module):
    def __init__(self, 
                 common_group=True,
                 common_groups=1,
                 vis_dim=[i for i in [4,8,16,32]], 
                 ir_dim=[i for i in [4,8,16,32]], 
                 common_dim=[i for i in [4,8,16,32]], 
                 inner_dim1=[i for i in [8,16,16,32]],
                 inner_dim2=[i for i in [8,8,4,4]],
                 out_dim=[8,8,4,4]):
        super().__init__()
        self.edge_op = CannyDetector()
        self.visir_edge = SEdgeDetector(in_dim=2)

        self.get_amap = AttentionMap(in_dim=1, dim=[2,4,8])

        self.feat_extract = FeatureExtractor(vis_dim=vis_dim, 
                                             ir_dim=ir_dim, 
                                             common_dim=common_dim, 
                                             common_group=common_group, 
                                             common_groups=common_groups)

        self.restruct = Reconstructor(vis_dim=vis_dim, 
                                      ir_dim=ir_dim, 
                                      common_dim=common_dim, 
                                      inner_dim1=inner_dim1, 
                                      inner_dim2=inner_dim2, 
                                      out_dim=out_dim)

        self.trans = MultiTrans(common_dim=common_dim, common_groups=common_groups)

        self.fusion = Fusion(sum(inner_dim2))

    def forward(self, data):
        vis = data['vi_y']
        ir = data['ir_y']

        visir_edge = self.visir_edge(torch.cat([vis, ir], dim=1), self.edge_op)

        
        priat_map, coat_map = self.get_amap(vis, ir)

        # use to generate feature map, in test 
        # im = (255 * vis * priat_map).detach().cpu().squeeze(0).squeeze(0)
        # from PIL import Image
        # im = Image.fromarray(im.numpy().astype('uint8'))
        # im.save('vip.png', quality=100)

        # im = (255 * ir * priat_map).detach().cpu().squeeze(0).squeeze(0)
        # from PIL import Image
        # im = Image.fromarray(im.numpy().astype('uint8'))
        # im.save('irp.png', quality=100)

        # im = (255 * vis * coat_map).detach().cpu().squeeze(0).squeeze(0)
        # from PIL import Image
        # im = Image.fromarray(im.numpy().astype('uint8'))
        # im.save('vic.png', quality=100)

        # im = (255 * ir * coat_map).detach().cpu().squeeze(0).squeeze(0)
        # from PIL import Image
        # im = Image.fromarray(im.numpy().astype('uint8'))
        # im.save('irc.png', quality=100)

        feat = self.feat_extract(vis*priat_map, ir*priat_map, vis*coat_map, ir*coat_map)
        vis_prifeat = feat['vis_prifeat']
        ir_prifeat = feat['ir_prifeat']
        vis_cofeat = feat['vis_cofeat']
        ir_cofeat = feat['ir_cofeat']

        common_feat = self.trans(vis_cofeat, ir_cofeat, ir)
        
        feat = self.restruct(vis_prifeat, ir_prifeat, common_feat, self.edge_op, visir_edge)

        fusion = self.fusion(feat)

        return {'ae_out': fusion.clamp(0, 1)}


if __name__ == '__main__':
    vis = torch.rand([2,1,300,400]).to('cuda')
    ir = torch.rand([2,1,300,400]).to('cuda')
    data= {'vis': vis, 'etc': ir}

    wnet = WNet().to('cuda')
    out = wnet(data)

    print(out['ae_out'].shape)
        
