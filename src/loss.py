import torch 
import torch.nn as nn
import torch.nn.functional as F
from metric import SSIM


class MSELoss(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        assert mode in ['none', 'mean', 'sum'], f'mode: {mode} is not supported !'
        self.loss_fn = nn.MSELoss(reduction=mode)

    def forward(self, im, imf, weight=1.):
        loss = self.loss_fn(im, imf)
        return weight * loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, down=0.3, up=0.7, mode='mean'):
        super().__init__()
        assert down>0. , f'down: {down} is not correct !'
        assert up<1., f'up: {up} is not correct !'
        assert mode in ['none', 'mean', 'sum'], f'mode: {mode} is not supported !'

        self.down = down
        self.up = up
        self.mode = mode
        self.ssim = SSIM(window_size, sigma)

    def forward(self, im, imf, weight=1., if_dynamic=False, down_gain=0.2, up_gain=2):
        DEV = im.device
        if if_dynamic:
            cond1 = torch.where(im<self.down, im*down_gain, torch.zeros([1], device=DEV))
            cond2 = torch.where(im>self.up, im*up_gain, torch.zeros([1], device=DEV))
            cond3 = (im>self.down) * (im<self.up)
            cond = cond1 + cond2 + weight * cond3
            ssim_loss = cond * (1 - self.ssim(im, imf)['ssim'])
        else:
            ssim_loss = weight * (1 - self.ssim(im, imf)['ssim'])

        if self.mode == 'mean': return ssim_loss.mean()
        elif self.mode == 'sum': return ssim_loss.sum()
        else: return ssim_loss


class GradLoss(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        assert mode in ['none', 'mean', 'sum'], f'mode: {mode} is not supported !'
        self.loss_fn = nn.SmoothL1Loss(reduction=mode)

    def gradient(self, x):
        DEV = x.device
        x_pad = F.pad(x, (1,1,1,1), 'replicate')

        with torch.no_grad():
            laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(DEV)
            return F.conv2d(x_pad, kernel, stride=1, padding=0)

    def forward(self, im, imf, weight=1.):
        loss = self.loss_fn(self.gradient(im), self.gradient(imf))
        return weight * loss


class StdLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im, im_fusion, alpha=0.5, beta=1., weight=1.):
        mean = (im.mean() - im_fusion.mean()).abs()
        std = (im.std() - im_fusion.std()).abs()
        loss = alpha * mean + beta * std

        return loss * weight

# add new loss, test
class SharpnessLoss(nn.Module):
    def __init__(self):
        super(SharpnessLoss, self).__init__()
    
    def forward(self, x):
        # 定义拉普拉斯算子
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        
        # 将滤波器移至相应的设备（CPU/GPU）
        laplacian_kernel = laplacian_kernel.to(x.device)
        
        # 计算图像的拉普拉斯映射
        laplacian_x = F.conv2d(x, laplacian_kernel, padding=1, groups=x.size(1))
        
        # 计算Sharpness Loss
        loss = -torch.mean(torch.abs(laplacian_x))
        return loss

class ContrastLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(ContrastLoss, self).__init__()
        self.lambda_weight = lambda_weight

    def smooth_histogram(self, x, bins=256, min=0, max=1, sigma=0.01):
        # 计算每个bin的中心点
        bin_centers = torch.linspace(min, max, bins).to(x.device)

        # 扩展bin_centers以匹配x的每个元素
        bin_centers = bin_centers.view(1, 1, -1)

        # 扩展x以匹配bin_centers的形状
        x = x.view(-1, 1, 1)

        # 计算高斯核
        diff = x - bin_centers
        hist = torch.exp(-0.5 * (diff / sigma) ** 2)

        # 对直方图进行归一化
        hist = hist / (hist.sum(2, keepdim=True) + 1e-8)

        # 求和所有图像的直方图
        hist = hist.sum(0).squeeze(0)
        return hist
    
    def forward(self, fused_image, ir_image, visible_gray, bins=256, min=0, max=1, sigma=0.01):
        # 计算平滑直方图
        hist_fused = self.smooth_histogram(fused_image, bins, min, max, sigma)
        hist_ir = self.smooth_histogram(ir_image, bins, min, max, sigma)
        hist_visible = self.smooth_histogram(visible_gray, bins, min, max, sigma)
        
        # 计算损失
        loss_ir = F.mse_loss(hist_fused, hist_ir)
        loss_visible = F.mse_loss(hist_fused, hist_visible)
        
        # 加权两个损失
        alpha = 0.5  # 权重可以根据需要调整
        total_loss = alpha * loss_ir + (1 - alpha) * loss_visible
        return total_loss

# class ContrastLoss(nn.Module):
#     def __init__(self, lambda_weight=1.0):
#         super(ContrastLoss, self).__init__()
#         self.lambda_weight = lambda_weight
    
#     def forward(self, x, y):
#         # 计算两个图像的均值和标准差
#         mean_x = torch.mean(x)
#         mean_y = torch.mean(y)
#         std_x = torch.std(x)
#         std_y = torch.std(y)
        
#         # 计算Contrast Loss
#         loss = torch.abs(mean_x - mean_y) + self.lambda_weight * torch.abs(std_x - std_y)
#         return loss
"""
how to use
# 实例化损失函数
sharpness_loss_fn = SharpnessLoss()
contrast_loss_fn = ContrastLoss(lambda_weight=0.5)

# 假设ir_image, visible_image, fused_image是输入的三张图像
# 它们的维度应为 (batch size x channels x height x width)
ir_image = torch.randn(8, 3, 256, 256)
visible_image = torch.randn(8, 3, 256, 256)
fused_image = torch.randn(8, 3, 256, 256)

# 计算各自的Sharpness Loss
sharpness_loss_ir = sharpness_loss_fn(ir_image)
sharpness_loss_visible = sharpness_loss_fn(visible_image)
sharpness_loss_fused = sharpness_loss_fn(fused_image)

# 计算总的Sharpness Loss
lambda1, lambda2 = 0.5, 0.5  # 可以调整这两个参数
total_sharpness_loss = sharpness_loss_fused - lambda1 * sharpness_loss_ir - lambda2 * sharpness_loss_visible

# 计算Contrast Loss
contrast_loss_ir_fused = contrast_loss_fn(fused_image, ir_image)
contrast_loss_visible_fused = contrast_loss_fn(fused_image, visible_image)

# 计算总的Contrast Loss
total_contrast_loss = contrast_loss_ir_fused + contrast_loss_visible_fused

# 输出损失值
print("Total Sharpness Loss:", total_sharpness_loss.item())
print("Total Contrast Loss:", total_contrast_loss.item())

"""