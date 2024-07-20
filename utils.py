import logging
import numpy as np
import torch
from PIL import Image
from scipy.misc import imsave
from torch.autograd import Variable
from einops.einops import rearrange


def logger_config(log_path, logging_name):
    '''
    :param log_path: output log path
    :param logging_name: name
    '''

    # get logger
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    # get the file log handle and set the log level
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)

    # generate and set the file log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # console: console output, handler: file output.
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # add handle
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def rgb_to_ycbcr(image, device):
    rgb_array = rearrange(image, '1 c h w ->1 h w c')

    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.169, -0.331, 0.5],
                                     [0.5, -0.419, -0.081]]).to(device)

    ycbcr_array = torch.matmul(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, :, 0]
    cb_channel = ycbcr_array[:, :, :, 1]
    cr_channel = ycbcr_array[:, :, :, 2]

    y_channel = y_channel.clamp(0.0, 255.0)
    return y_channel.unsqueeze(0), cb_channel.unsqueeze(0), cr_channel.unsqueeze(0)


def ycbcr_to_rgb(y, cb, cr, device):
    ycbcr_array = torch.cat([y, cb, cr], dim=1)
    ycbcr_array = rearrange(ycbcr_array, '1 c h w ->1 h w c')

    transform_matrix = torch.tensor([[1, 0, 1.402],
                                     [1, -0.344136, -0.714136],
                                     [1, 1.772, 0]]).to(device)

    rgb_array = torch.matmul(ycbcr_array, transform_matrix.T)

    rgb_array = torch.clamp(rgb_array, 0, 1.0)
    rgb_array = rearrange(rgb_array, 'b h w c-> b c h w')

    return rgb_array


# tensor to PIL Image
def tensor2img(img, is_norm=True):
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    return img.astype(np.uint8)


def save_img(img, name, is_norm=True):
    img = tensor2img(img, is_norm=True)
    img = Image.fromarray(img)
    img.save(name)


def randrot(img):
    mode = np.random.randint(0, 4)
    return rot(img, mode)


def randfilp(img):
    mode = np.random.randint(0, 3)
    return flip(img, mode)


def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img


def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img


def cc(A, B, F):
    img1 = torch.squeeze(A)
    img2 = torch.squeeze(B)
    fuse = torch.squeeze(F)
    batch, _, _ = img1.shape
    c = 0
    for i in range(batch):
        A = img1[i] * 255
        B = img2[i] * 255
        F = fuse[i] * 255
        rAF = torch.sum((A - torch.mean(A)) * (F - torch.mean(F))) / torch.sqrt(
            torch.sum((A - torch.mean(A)) ** 2) * torch.sum((F - torch.mean(F)) ** 2))
        rBF = torch.sum((B - torch.mean(B)) * (F - torch.mean(F))) / torch.sqrt(
            torch.sum((B - torch.mean(B)) ** 2) * torch.sum((F - torch.mean(F)) ** 2))
        c += (rAF + rBF) / 2
    c = c / batch
    return c


def agm(img_ir, img_vi, fuse, I1, I2, cnt1):
    fuse_copy = Variable(fuse.data.clone(), requires_grad=False)
    I1_copy = Variable(I1.data.clone(), requires_grad=False)
    I2_copy = Variable(I2.data.clone(), requires_grad=False)
    img_ir_copy = Variable(img_ir.data.clone(), requires_grad=False)
    img_vi_copy = Variable(img_vi.data.clone(), requires_grad=False)

    w1 = cc(img_ir_copy, img_vi_copy, fuse_copy)
    w2 = cc(img_ir_copy, img_vi_copy, I1_copy)
    w3 = cc(img_ir_copy, img_vi_copy, I2_copy)
    if torch.isnan(w3):
        w3 = torch.tensor(0).cuda()
    # avoid negative number
    if w1 < 0 or w2 < 0 or w3 <= 0:
        w1 = w1 + 1
        w2 = w2 + 1
        w3 = w3 + 1
    if w3 > w2:
        I1 = I2
        w2 = w3
        cnt1 += 1

    # the adaptive weight and guidance image
    img_guidance = I1
    w = 3 * w2 / w1
    # choose better result
    flag = w1 >= w3
    return img_guidance, w.item(), flag, cnt1


def save_choose_best(fuse, data_path, img_name, e, flag, cnt2):
    fuse_copy = fuse.cpu().detach().numpy()
    if e == 0:
        for j, name in enumerate(img_name):
            img = fuse_copy[j, 0, :, :] * 255
            imsave(data_path + "/img2/" + name, img)
    else:
        if flag:
            cnt2 += 1
            for j, name in enumerate(img_name):
                img = fuse_copy[j, 0, :, :] * 255
                imsave(data_path + "/img2/" + name, img)
    return cnt2


def resume(model, optimizer=None, model_save_path=None, device=None, is_train=True):
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if is_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']
        total_it = checkpoint['total_it']
        return model, optimizer, ep, total_it
    else:
        return model


def save_model(model_name, e, total_it, model, optimizer, device):
    model.eval()
    model = model.cpu()
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': e,
             'total_it': total_it}

    torch.save(state, model_name)
    model.train()
    model.to(device)
