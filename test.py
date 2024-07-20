import argparse
from tqdm import tqdm
from create_dataset import *
from net import AGMFusion
from utils import *

# TODO - 计算模型的推理时间
def calcTime():

    import numpy as np
    from torchvision.models import resnet50
    import torch
    from torch.backends import cudnn
    import tqdm
    from thop import profile

    '''  导入你的模型
    from module.amsnet import amsnet, anet, msnet, iresnet18, anet2, iresnet2, amsnet2
    from module.resnet import resnet18, resnet34
    from module.alexnet import AlexNet
    from module.vgg import vgg
    from module.lenet import LeNet
    from module.googLenet import GoogLeNet
    from module.ivgg import iVGG
    '''


    cudnn.benchmark = True

    device = 'cuda:0'
    # model.load_state_dict(torch.load('D:\code\MUFusion-github\model\\Epoch_25.model', map_location='cuda:0'))
    repetitions = 1000

    dummy_input = torch.rand(1, 3, 480, 640).to(device)



    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(30):
            y, cb, cr = rgb_to_ycbcr(dummy_input, device)
            y = y.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            _ = ycbcr_to_rgb(y, cb, cr, device)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            y, cb, cr = rgb_to_ycbcr(dummy_input, device)
            # y = y.to(device)
            # cb = cb.to(device)
            # cr = cr.to(device)
            _ = ycbcr_to_rgb(y, cb, cr, device)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))

def load_model(path, opts):
    device = torch.device("cuda:{}".format(opts.gpu_id) if torch.cuda.is_available() else "cpu")
    model = AGMFusion(2)
    model = model.to(device)

    # if error please try another way to load model
    # way 1:
    # checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(checkpoint['model'], False)
    # way 2:
    model.load_state_dict(torch.load(path, map_location='cuda:0'))

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para / 1000))

    # this sentence is very important!
    model.eval()

    return model


# visible image-RGB
def fuse(model, output_dir, opts):
    device = torch.device("cuda:{}".format(opts.gpu_id) if torch.cuda.is_available() else "cpu")

    # define dataset
    test_dataset = FusionData(opts.test_path)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=False,)
    test_bar = tqdm(test_loader)

    with torch.no_grad():
        for it, (img_ir, img_vi, img_names) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            y, cb, cr = rgb_to_ycbcr(img_vi, device)
            y = y.to(device)
            cb = cb.to(device)
            cr = cr.to(device)

            fused_y = model(torch.cat([y, img_ir], 1))
            fused_img = ycbcr_to_rgb(fused_y, cb, cr, device)

            for i in range(len(img_names)):
                img_name = img_names[i]
                fusion_save_name = os.path.join(output_dir, img_name)
                save_img(fused_img[i, ::], fusion_save_name)
                test_bar.set_description('Image: {} '.format(img_name))


# visible image-L
def fuse_gray(model, output_dir, opts):
    device = torch.device("cuda:{}".format(opts.gpu_id) if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = FusionDataGray(opts.test_path)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=False,)
    test_bar = tqdm(test_loader)

    with torch.no_grad():
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)

            fused_y = model(torch.cat([img_vi, img_ir], 1))

            for i in range(len(img_names)):
                img_name = img_names[i]
                fusion_save_name = os.path.join(output_dir, img_name)
                save_img(fused_y[i, ::], fusion_save_name)
                test_bar.set_description('Image: {} '.format(img_name))


def main():
    opts = parse_opt()

    # load some models and fuse images
    for x in range(len(opts.model_paths)):
        output_dir = opts.output_path + 'Epoch_' + str(opts.model_paths[x]) + '/'
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        model_name = opts.model_path + 'Epoch_' + str(opts.model_paths[x]) + '.pth'
        model = load_model(model_name, opts)

        if opts.mode == 'RGB':
            fuse(model, output_dir, opts)
        elif opts.mode == 'L':
            fuse_gray(model, output_dir, opts)
        else:
            print('mode error')


def parse_opt():
    parser = argparse.ArgumentParser()

    # data loader related
    # parser.add_argument('--test_path', type=str,
    #                     default='D:\code\Evaluation-for-Image-Fusion-main\Image\Source-Image\MS2\\',
    #                     help='path of data')
    parser.add_argument('--test_path', type=str, default='./dataset/test/MSRS/', help='path of data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='# of threads for data loader')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    # test related
    parser.add_argument('--mode', type=str, default='RGB', help='vi img type, i.e. L or RGB')
    parser.add_argument('--model_paths', type=list, default=[50],
                        help='specified the name of saved models for testing, i.e., [4,2] means using Epoch_4.pth and Epoch_2.pth')
    parser.add_argument('--model_path', type=str, default='./model/',
                        help='path for saving result images and models')
    parser.add_argument('--output_path', type=str, default='./fusion_result/',
                        help='path for saving result images and models')

    opt = parser.parse_args()
    args = vars(opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
        print('%s: %s' % (str(name), str(value)))
    return opt


if __name__ == '__main__':
    #calcTime()
    main()
