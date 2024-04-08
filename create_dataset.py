from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import os
import torch
from utils import randrot, randfilp
from natsort import natsorted


class TrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, data_path, crop=lambda x: x):
        super(TrainData, self).__init__()

        self.vis_folder = os.path.join(data_path, 'vi')
        self.ir_folder = os.path.join(data_path, 'ir')
        self.I1_folder = os.path.join(data_path, 'img1')
        self.I2_folder = os.path.join(data_path, 'img2')

        # gain infrared and visible images list
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        # self.ir_list = self.ir_list[13000:25000]
        print('train images:', len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        i1_path = os.path.join(self.I1_folder, image_name)
        i2_path = os.path.join(self.I2_folder, image_name)

        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        i1 = self.imread(path=i1_path)
        i2 = self.imread(path=i2_path)

        # data augment, including flipping, rotating, and random cropping
        # vis_ir = torch.cat([vis, ir, i1, i2], dim=1)
        # if vis_ir.shape[-1] <= 128 or vis_ir.shape[-2] <= 128:
        #     vis_ir = TF.resize(vis_ir, 128)
        # vis_ir = randfilp(vis_ir)
        # vis_ir = randrot(vis_ir)
        # patch = self.crop(vis_ir)

        # vis, ir, i1, i2 = torch.split(patch, [1, 1, 1, 1], dim=1)
        # vis, ir, i1, i2 = torch.split(vis_ir, [1, 1, 1, 1], dim=1)

        return ir.squeeze(0), vis.squeeze(0), i1.squeeze(0), i2.squeeze(0), image_name

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path):
        img = Image.open(path).convert('L')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class FusionData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, data_path):
        super(FusionData, self).__init__()
        self.vis_folder = os.path.join(data_path, 'vi')
        self.ir_folder = os.path.join(data_path, 'ir')

        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)

        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path, vis_flage=False)
        return ir, vis, image_name

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path, vis_flage=True):

        if vis_flage:  # visible images; RGB channel
            img = Image.open(path).convert('RGB')
            im_ts = TF.to_tensor(img)
        else:  # infrared images single channel
            img = Image.open(path).convert('L')
            im_ts = TF.to_tensor(img)
        return im_ts


# visible images: single channel
class FusionDataGray(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, data_path):
        super(FusionDataGray, self).__init__()
        self.vis_folder = os.path.join(data_path, 'vi')
        self.ir_folder = os.path.join(data_path, 'ir')

        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)

        # read image as type Tensor
        vis, w, h = self.imread(path=vis_path)
        ir, w, h = self.imread(path=ir_path)
        return ir.squeeze(0), vis.squeeze(0), image_name, w, h

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path):
        img = Image.open(path).convert('L')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts

