import torch.nn.functional as F
import torch
import pytorch_msssim


def sf_loss(fused_result, input_ir, device):
    loss = torch.norm(sf(fused_result, device) - sf(input_ir, device))

    return loss


def sf(f1, device, kernel_radius=5):
    b, c, h, w = f1.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
        .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
        .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)

    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)

    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2

    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    return 1 - f1_sf


def fusion_loss(e, ir, vi, fuse, gd, w, device):
    msssim = pytorch_msssim.msssim

    sf_loss1 = 0.00001 * sf_loss(fuse, vi, device)
    sf_loss2 = 0.00001 * sf_loss(fuse, ir, device)
    sf_loss3 = 0.00001 * sf_loss(fuse, gd, device)

    ssim_loss1 = 1 - msssim(fuse, vi, normalize=True)
    ssim_loss2 = 1 - msssim(fuse, ir, normalize=True)
    ssim_loss3 = 1 - msssim(fuse, gd, normalize=True)

    loss_content = 1 * ssim_loss1 + 1 * ssim_loss2 + 1 * sf_loss1 + 1 * sf_loss2
    if e == 0:
        loss = loss_content
        return loss, loss_content, torch.tensor(0), ssim_loss1 + ssim_loss2, sf_loss1 + sf_loss2, torch.tensor(0), torch.tensor(0)
    else:
        loss_guidance = 1 * ssim_loss3 + 1 * sf_loss3
        loss = loss_content + w * loss_guidance
        return loss, loss_content, loss_guidance, ssim_loss1 + ssim_loss2, sf_loss1 + sf_loss2, ssim_loss3, sf_loss3
