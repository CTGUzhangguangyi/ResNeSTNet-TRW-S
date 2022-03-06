import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/my_res++-84.pth')
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        data_path = './data/TestDataset/{}/'.format('Kvasir')
        opt = parser.parse_args()
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        for name, module in self.submodule._modules.items():
            # if "fc" in name:
            #     x = x.view(x.size(0), -1)
            image, gt, name1 = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            module.eval()
            print(image.shape)
            x = module(image)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x



        ################修改成自己的网络，直接在network.py中return你想输出的层

        # Conv2d-1, = self.submodule(
        #     x)
        # outputs["Conv2d-1"] = Conv2d1
        # return outputs
        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    pic_dir = './data/TestDataset/Kvasir/images/cju0u82z3cuma0835wlxrnrjv.png'  # 往网络里输入一张图片
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)

    img = img.cuda()
    net =smp.Unet(
        # encoder_name="efficientnet-b3",
        # encoder_name="resnext50_32x4d",
        # encoder_name="mobilenet_v2",
        # encoder_name="timm-mobilenetv3_large_100",
        # encoder_name="timm-gernet_l",
        # encoder_name="xception",
        # encoder_name="timm-res2net50_14w_8s",
        encoder_name="timm-resnest50d_4s2x40d",
        # encoder_name="timm-skresnet34",
        encoder_weights="imagenet",
        in_channels=3,
        # encoder_depth=5,
        classes=1,
        # activation='sigmoid'
        # aux_params=aug_params,
    ).cuda()
    net.load_state_dict(torch.load('./snapshots/PraNet_Res2Net/my_rest1-29.pth'))
    # net.to(device)
    exact_list = None
    # exact_list = ['conv1_block',""]
    dst = './features'  # 保存的路径
    therd_size = 256  # 有些图太小，会放大到这个尺寸

    myexactor = FeatureExtractor(net, exact_list)

    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)


if __name__ == '__main__':
    get_feature()
