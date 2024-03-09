import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv_bn_relu(ch_in, ch_out, kernel=3, stride=1, padding=0, 
                 bn=True, relu=True, group=False, groups=4, padding_mode='replicate'):
    assert (kernel % 2) == 1, f'only odd kernel is supported but {kernel}'
    assert padding_mode in ['reflect', 'replicate', 'zeros'], f'{padding_mode} is not supported'

    groups = groups if group else 1

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, 
                            groups=groups, bias=not bn, padding_mode=padding_mode))
    if bn: layers.append(nn.BatchNorm2d(ch_out))
    if relu: layers.append(nn.CELU())
    layers = nn.Sequential(*layers)

    return layers


def conv1d_bn_relu(ch_in, ch_out, kernel=3, stride=1, padding=0, 
                   bn=True, relu=True, group=False, groups=4, padding_mode='replicate'):
    assert (kernel % 2) == 1, f'only odd kernel is supported but kernel={kernel}'
    assert padding_mode in ['reflect', 'replicate', 'zeros'], f'{padding_mode} is not supported'

    groups = groups if group else 1

    layers = []
    layers.append(nn.Conv1d(ch_in, ch_out, kernel, stride, padding, 
                            groups=groups, bias=not bn, padding_mode=padding_mode))
    if bn: layers.append(nn.BatchNorm1d(ch_out))
    if relu: layers.append(nn.CELU())
    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0, 
                  bn=True, relu=True, group=False, groups=4):
    assert (kernel % 2) == 1, f'only odd kernel is supported but kernel={kernel}'
    
    groups = groups if group else 1

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding, 
                                     output_padding, groups=groups, bias=not bn))
    if bn: layers.append(nn.BatchNorm2d(ch_out))
    if relu: layers.append(nn.CELU())
    layers = nn.Sequential(*layers)

    return layers


def get_denset121(if_pretrain=True)->dict:
    densenet = torchvision.models.densenet121(pretrained=if_pretrain)

    # output: 64 -> 1/2, 128
    layer1 = nn.Sequential(densenet.features[4], densenet.features[5])
    # output: 128 -> 1/4, 256
    layer2 = nn.Sequential(densenet.features[6], densenet.features[7])
    # output: 256 -> 1/8, 512
    layer3 = nn.Sequential(densenet.features[8], densenet.features[9])
    # output: 512 -> 1/8, 1024
    layer4 = nn.Sequential(densenet.features[10], densenet.features[11])

    return {'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4}


def get_resnet34(if_pretrain=True)->dict:
    resnet = torchvision.models.resnet34(pretrained=if_pretrain)

    # output: 64 -> 1, 64
    layer1 = resnet.layer1
    # output: 64 -> 1/2, 128
    layer2 = resnet.layer2
    # output: 128 -> 1/4, 256
    layer3 = resnet.layer3
    # output: 256 -> 1/8, 512
    layer4 = resnet.layer4

    return {'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4}


def get_resnet18(if_pretrain=True)->dict:
    resnet = torchvision.models.resnet18(pretrained=if_pretrain)

    # output: 64 -> 1, 64
    layer1 = resnet.layer1
    # output: 64 -> 1/2, 128
    layer2 = resnet.layer2
    # output: 128 -> 1/4, 256
    layer3 = resnet.layer3
    # output: 256 -> 1/8, 512
    layer4 = resnet.layer4

    return {'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4}


def calc_mean_std(feat, eps=1e-5):
    size = feat.shape
    B, C = size[:2]
    assert (len(size) == 4), 'tensor must have 4 dim'
    
    feat_var = feat.view(B, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(B, C, 1, 1)
    feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)

    return feat_mean, feat_std


def adaptive_insnorm(content_feat, style_feat):
    assert (content_feat.shape[:2] == style_feat.shape[:2]), \
        'sizes of channel and batch must be equal'

    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / content_std

    return normalized_feat * style_std + style_mean


class Hister():
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

    def process(self, t4d):
        _, cdf = self.get_pdf_cdf(t4d)
        histed = self.hist_filter(t4d, cdf)

        hist_w = 0.75*(-torch.std(t4d.flatten(1,-1), dim=-1, keepdim=True)).exp()
        # hist_w = 0.5*(torch.std(t4d.flatten(1,-1), dim=-1, keepdim=True))

        compound = (1-hist_w) * histed + hist_w * t4d

        return compound
