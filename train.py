import argparse
import datetime
import time
import sys
from torch.optim import Adam
from create_dataset import *
from losses import *
from net import AGMFusion
from utils import *


def main():
    train_dataset = TrainData(opts.data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True)

    # each epoch iteration
    ep_iter = len(train_loader)
    max_iter = opts.epoch * ep_iter
    print('Training iter: {}'.format(max_iter))

    device = torch.device("cuda:{}".format(opts.gpu_id) if torch.cuda.is_available() else "cpu")
    model = AGMFusion(2).to(device)

    optimizer = Adam(model.parameters(), opts.lr)

    if opts.is_resume:
        checkpoint = torch.load(opts.model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], False)
        start_ep = checkpoint['epoch']
        total_it = checkpoint['total_it']

    else:
        start_ep = -1
        total_it = 0
    start_ep += 1

    # log
    log_dir = os.path.join('./logs', 'logger', opts.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Epoch')
    logger.info('Parameter: {:.6f}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    log_dir2 = os.path.join('./logs', 'logger', opts.name)
    os.makedirs(log_dir2, exist_ok=True)
    log_path2 = os.path.join(log_dir2, 'log2.txt')
    if os.path.exists(log_path2):
        os.remove(log_path2)
    logger2 = logger_config(log_path=log_path2, logging_name='Iter')

    best_loss = 10
    start = train_start = time.time()

    # begin training
    for e in range(start_ep, opts.epoch):
        model.train()

        cnt1 = 0  # Count the number of times I2 is used in each epoch
        cnt2 = 0  # Count the number of times the current result is saved in each epoch

        for it, (img_ir, img_vi, I1, I2, img_name) in enumerate(train_loader):
            # check I2 in the first epoch
            # if e == 0:
            #     tmp = torch.ones([batch_size, 1, 128, 128])
            #     if not tmp.equal(I2):
            #         print('img2 is not white')
            #         sys.exit()
            total_it += 1
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            # SOTA fused image
            I1 = I1.to(device)
            # best output
            I2 = I2.to(device)

            fuse = model(torch.cat([img_vi, img_ir], 1))
            optimizer.zero_grad()

            # the adaptive guidance model
            img_gd, w, flag, cnt1 = agm(img_ir, img_vi, fuse, I1, I2, cnt1)

            loss, loss_content, loss_guidance, msssim_loss, sf_loss1, ssim_gd, sf_gd = fusion_loss(e, img_ir, img_vi,
                                                                                                   fuse,
                                                                                                   img_gd, w, device)
            loss.backward()
            optimizer.step()

            # save outputs and choose best fused image for subsequent epoch.
            cnt2 = save_choose_best(fuse, opts.data_path, img_name, e, flag, cnt2)
            if it % opts.freq == 0:
                logger2.info(
                    'batch: [{}/{}], loss: {:.4f}, loss_content: {:.4f}, '
                    'loss_guidance: {:.4f}, msssim_loss: {:.4f},'
                    ' sf_loss: {:.4f}, ssim_gd: {:.4f}, sf_gd: {:.4f}, w: {:.4f}, cnt1: {}, cnt2: {}'.format(
                        it, ep_iter, loss.item(), loss_content.item(), loss_guidance.item(),
                        msssim_loss.item(), sf_loss1.item(), ssim_gd.item(), sf_gd.item(), w, cnt1, cnt2))

            # save model which has the least loss
            if loss.item() < best_loss and e > 0:
                best_loss = loss.item()
                filename = "Epoch_" + str(e) + "_" + \
                           "loss_" + str(round(best_loss, 4)) + ".pth"
                save_model_path = os.path.join(opts.model_path, filename)
                save_model(save_model_path, e, total_it, model, optimizer, device)

        end = time.time()
        training_time, glob_time = end - start, end - train_start
        now_it = total_it + 1
        eta = int((opts.epoch * len(train_loader) - now_it) * (glob_time / (now_it)))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info(
            'ep: [{}/{}], learning rate: {:.6f}, time consuming: {:.2f}s, loss: {:.4f}, loss_content: {:.4f}, loss_guidance: {:.4f}, msssim_loss: {:.4f},'
            ' sf_loss: {:.4f}, ssim_gd: {:.4f}, sf_gd: {:.4f}, w: {:.4f}, cnt1: {}, cnt2: {} Eta: {}\n'.format(
                e, opts.epoch, opts.lr, training_time, loss.item(), loss_content.item(), loss_guidance.item(),
                msssim_loss.item(), sf_loss1.item(), ssim_gd.item(), sf_gd.item(), w, cnt1, cnt2, eta))
        start = time.time()

        # save model
        save_model_path = os.path.join(opts.model_path, "Epoch_" + str(e) + ".pth")
        save_model(save_model_path, e, total_it, model, optimizer, device)


def parse_opt():
    parser = argparse.ArgumentParser()

    # data loader related
    parser.add_argument('--data_path', type=str, default='./dataset/train/', help='path of data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='# of threads for data loader')

    # training related
    parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate for training model')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--is_resume', type=str, default=None,
                        help='specified the dir of saved models for resume the training')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    # ouptput related
    parser.add_argument('--name', type=str, default='AGMFusion', help='folder name to save outputs')
    parser.add_argument('--model_path', type=str, default='./model/',
                        help='path for saving result images and models')
    parser.add_argument('--freq', type=int, default=10, help='freq (iteration) of display')
    opt = parser.parse_args()
    args = vars(opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
        print('%s: %s' % (str(name), str(value)))
    return opt


if __name__ == '__main__':
    opts = parse_opt()
    main()
