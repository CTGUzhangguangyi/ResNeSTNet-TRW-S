import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import *
from utils.dataloader import test_dataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/my_rest2-29.pth')
def horizontal_flip(image):
    # print(image.shape)
    h_image = image.cpu().detach().numpy().copy()
    # print(h_image)
    h_image = h_image[:,:, :,::-1 ]
    h_image = torch.from_numpy(h_image.copy())
    return h_image

def vertical_flip(image):
    v_image = image.cpu().detach().numpy().copy()
    v_image = v_image[ :,:,::-1, :]
    v_image = torch.from_numpy(v_image.copy())
    return v_image

def tta(model,image):
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_image = n_image.cuda()
    h_image = h_image.cuda()
    v_image = v_image.cuda()

    n_mask = model(n_image)
    h_mask = model(h_image)
    v_mask = model(v_image)

    n_mask = n_mask
    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)

    n_mask = n_mask.cuda()
    h_mask = h_mask.cuda()
    v_mask = v_mask.cuda()

    mean_mask = (n_mask + h_mask + v_mask) / 3.0

    return mean_mask
featers = {}
def get_hook(name):
    def hook(model, input, output):
        featers[name] = output.detach()
    return hook

def draw_features(width, height, channels,x,savename):
    '''
    x:        输入的array,某一层的网络层输出
    savename: 特征可视化的保存路径
    width, height: 分别表示可视化子图的个数,二者乘积等于channels
    '''
    fig = plt.figure(figsize=(32,32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(channels):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
#         print("{}/{}".format(i, channels))
#     plt.show()
    fig.savefig(savename, dpi=300)
    fig.clf()
    plt.close()
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)
    model = checkpoint['models']  # 提取网络结构
    # print(model)
    # model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
# for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
for _data_name in ['Kvasir1']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/512/{}/'.format(_data_name)
    opt = parser.parse_args()

    model = smp.Unet(
        # encoder_name="efficientnet-b3",
        # encoder_name="resnext50_32x4d",
        # encoder_name="mobilenet_v2",
        # encoder_name="timm-mobilenetv3_large_100",

        # encoder_name="xception",
        # encoder_weights="imagenet",
        # encoder_name="timm-gernet_l",
        # encoder_name="timm-res2next50",
        encoder_name="timm-resnest50d_4s2x40d",

        in_channels=3,
        # encoder_depth=5,
        classes=1,
        # activation='sigmoid'
        # aux_params=aug_params,
    )
    model = model.cuda()
    # model = xception_mseg()
    # model = ResUnetPlusPlus(3).cuda()

    model.load_state_dict(torch.load(opt.pth_path))
    # model.avgpool.register_forward_hook(get_hook('rfb2_1'))
    model.encoder.layer2[3].act3.register_forward_hook(get_hook('act3'))

    # print(model.encoder.layer4[2])
    model.cuda()
    model.eval()
    # print( model.decoder.rfb2_1)
    # print(model.)
    # for a,b in model.named_modules():
    #     print(a)

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # print(image.shape)
        res=model(image)
        print(featers['act3'].shape)
        abc = featers['act3'].cpu().numpy()
        draw_features(23, 23, 512, abc, os.path.join(save_path,name))
        break
        # # res = tta(model, image)
        # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # for i in range(len(res)):
        #     for c in range(1):
        #         cv2.imwrite(os.path.join(save_path,name), (res[i, c] * 255).astype('uint8'))