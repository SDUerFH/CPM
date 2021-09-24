import os
import scipy.io
import numpy as np
import skimage.transform
import glob
import torch
from torch.utils.data import Dataset
import scipy.misc
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from PIL import Image
import imageio

lms = None
imagefiles = None
weight = None

'''
1. Right ankle
2. Right knee
3. Right hip
4. Left hip
5. Left knee
6. Left ankle
7. Right wrist
8. Right elbow
9. Right shoulde
10. Left shoulder
11. Left elbow
12. Left wrist
13. Neck
14. Head top
'''


class LSPDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase_train=True,
                 weighted_loss=False, bandwidth=50):
        self.scaled_h = 368
        self.scaled_w = 368
        self.map_h = 45
        self.map_w = 45
        self.guassian_sigma = 21
        self.num_keypoints = 14
        self.num_train = 11000
        global lms, imagefiles, weight
        if lms is None or imagefiles is None or weight is None:
            # mat_lsp.shape (3, 14, 2000)
            mat_lsp = scipy.io.loadmat(os.path.join(root_dir, 'lsp_dataset/joints.mat'),
                                       squeeze_me=True, struct_as_record=False)['joints']
            # mat_lspet.shape (14, 3, 10000)
            mat_lspet = scipy.io.loadmat(os.path.join(root_dir, 'lspet_dataset/joints.mat'),
                                         squeeze_me=True, struct_as_record=False)['joints']
            # get the list of all the images' root
            image_lsp = np.array(glob.glob(os.path.join(root_dir,
                                                        'lsp_dataset/images/*.jpg'), recursive=True))
            image_lspet = np.array(glob.glob(os.path.join(root_dir,
                                                          'lspet_dataset/images/*.jpg'), recursive=True))
            # 正则匹配出来每张照片的编号,从0001到2000
            # 需要注意这里面的所有编号都是float型的
            # change: / -> \\
            image_nums_lsp = np.array([float(s.rsplit('\\')[-1][2:-4]) for s in image_lsp])
            image_nums_lspet = np.array([float(s.rsplit('\\')[-1][2:-4]) for s in image_lspet])
            # sort by img name
            sorted_image_lsp = image_lsp[np.argsort(image_nums_lsp)]
            sorted_image_lspet = image_lspet[np.argsort(image_nums_lspet)]
            # lms 与 imagefiles 中都是lspet在前, lsp在后
            # mat_lspet.transpose([2, 1, 0])[:, :2, :].shape 为 (10000, 2, 14)
            # mat_lsp.transpose([2, 0, 1])[:, :2, :].shape 为 (2000, 2, 14)
            # self.lms.shape 为 (12000, 2, 14)
            self.lms = np.append(mat_lspet.transpose([2, 1, 0])[:, :2, :],
                                 # only the x, y coords, not the "block or not" channel
                                 mat_lsp.transpose([2, 0, 1])[:, :2, :],
                                 axis=0)
            # (10000,) + (2000,) = (12000,)
            self.imagefiles = np.append(sorted_image_lspet, sorted_image_lsp)
            imgs_shape = []  # save the shape of every image, (12000, 2)
            for img_file in self.imagefiles:
                imgs_shape.append(Image.open(img_file).size)
            # np.array(imgs_shape)[:, :, np.newaxis].shape 为 (12000, 2, 1)
            # (12000, 2, 14) / (12000, 2, 1) = (12000, 2, 14)
            # lms_scaled 为标准化后的坐标
            lms_scaled = self.lms / np.array(imgs_shape)[:, :, np.newaxis]
            # 去除错误的数据,只有(0,1] 之间才是正确的数据(这里转成了float不知道有啥用),(12000, 2, 14)
            self.weight = np.logical_and(lms_scaled > 0, lms_scaled <= 1).astype(np.float32)
            # x,y相乘,只有值为1.0才说明这组数据没问题
            # (12000, 2, 14) -> (12000, 14)
            self.weight = self.weight[:, 0, :] * self.weight[:, 1, :]
            # (12000, 14) -> (12000, 15)
            self.weight = np.append(self.weight, np.ones((self.weight.shape[0], 1)), axis=1)
            # (12000, 1, 15) -> (12000, 6, 15)
            self.weight = self.weight[:, np.newaxis, :].repeat(6, axis=1)
            # 下面是作者自己所做的创新
            # weighted loss一般用作图像分割领域，尤其是在分割的边缘像素和非边缘像素样本数量不均匀，差别过大，
            # 导致的loss影响，学习较慢，精确率较低的问题。举个例子来说，比如一共100个像素，边缘像素有2个，非边缘有98个，
            # 那么边缘损失的权重为0.98，非边缘损失的权重为0.02，给数量少的正样本足够大的权重，卷积网络就可训练出最后满意结果。
            if weighted_loss and phase_train:
                # self.num_train 为 11000
                # (11000, 2, 14) -> (11000, 28)
                datas = lms_scaled[:self.num_train].reshape(self.num_train, -1)
                # 将噪声点的值都设置为0
                datas[datas < 0] = 0
                datas[datas > 1] = 0
                # 将数据从 28维 降低至 3维
                datas_pca = PCA(n_components=3).fit_transform(datas)
                # 核密度估计(也称作Parzen窗)
                # bandwidth 为预定义的50
                kde = KernelDensity(bandwidth=bandwidth).fit(datas_pca)
                # score_samples返回的是点x对应概率的log值,要使用exp求指数还
                p = np.exp(kde.score_samples(datas_pca))
                # 中位数
                p_median = np.median(p)
                p_weighted = p_median / p
                self.weight[:self.num_train] *= p_weighted[:, np.newaxis, np.newaxis]
            # 注意 lms, imagefiles, weight 都是全局变量
            lms = self.lms
            imagefiles = self.imagefiles
            weight = self.weight
        else:
            self.lms = lms
            self.imagefiles = imagefiles
            self.weight = weight

        self.transform = transform
        self.phase_train = phase_train

    def __len__(self):
        # 训练阶段,11000
        if self.phase_train:
            return self.num_train
        else:
            # 测试阶段,1000
            return self.imagefiles.shape[0] - self.num_train

    def __getitem__(self, idx):
        # 非训练阶段返回的idx 为实际的 idx + self.num_train
        if not self.phase_train:
            idx += self.num_train
        image = imageio.imread(self.imagefiles[idx])  # 读取图片
        image_h, image_w = image.shape[:2]  # 获取维度
        # 这里lms取到的是正常坐标,不是归一化后的
        lm = self.lms[idx].copy()  # (2, 14)
        # 将图片各个坐标缩放到 45x45 像素对应的位置
        lm[0] = lm[0] * self.map_w / image_w
        lm[1] = lm[1] * self.map_h / image_h
        gt_map = []
        for (x, y) in zip(lm[0], lm[1]):
            if x > 0 and y > 0:
                # 将坐标转化成了以(x,y)为极值点,σ为参数的二维高斯分布,极值为1
                # 用图像来表示的话就会是以关节坐标为中心,向外颜色逐渐降低的一个热图,这里的σ是超参数
                heat_map = guassian_kernel(self.map_w, self.map_h, x, y, self.guassian_sigma)  # σ为 21
            else:
                heat_map = np.zeros((self.map_h, self.map_w))
            gt_map.append(heat_map)
        # 将每一组坐标都化成一个二维高斯分布,也就是一个热图
        gt_map = np.array(gt_map)  # (14, 45, 45)
        # 45x45的全1数组 - gt_map第一维最大值(45x45) = (45, 45)
        gt_backg = np.ones([self.map_h, self.map_w]) - np.max(gt_map, 0)  # (45, 45)
        # (15, 45, 45) = (14, 45, 45) + (1, 45, 45)
        gt_map = np.append(gt_map, gt_backg[np.newaxis, :, :], axis=0)

        # 所有坐标的中值,躯干中心
        center_x = (self.lms[idx][0][self.lms[idx][0] < image_w].max() +
                    self.lms[idx][0][self.lms[idx][0] > 0].min()) / 2
        center_y = (self.lms[idx][1][self.lms[idx][1] < image_h].max() +
                    self.lms[idx][1][self.lms[idx][1] > 0].min()) / 2

        center_x = center_x / image_w * self.scaled_w
        center_y = center_y / image_h * self.scaled_h

        # 躯干中心的高斯核
        center_map = guassian_kernel(self.scaled_w, self.scaled_h,
                                     center_x, center_y, self.guassian_sigma)
        weight = self.weight[idx]
        # image是图片的原始数据, gt_map是(15, 45, 45)的数据,
        # center_map 是(45, 45)的数据, weight是作者自己加的数据
        sample = {'image': image, 'gt_map': gt_map, 'center_map': center_map, 'weight': weight}

        if self.transform:
            sample = self.transform(sample)

        return sample


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
    """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new.astype(np.float32), 'gt_map': sample['gt_map'],
                'center_map': sample['center_map'], 'weight': sample['weight']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, phase_cuda=False):
        self.phase_cuda = phase_cuda

    def __call__(self, sample):
        image, gt_map, center_map, weight = sample['image'], sample['gt_map'], \
                                            sample['center_map'], sample['weight']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'gt_map': torch.from_numpy(gt_map).float(),
                'center_map': torch.from_numpy(center_map).float(),
                'weight': torch.from_numpy(weight).float()}


class Scale(object):
    def __init__(self, height, weight):
        self.height = height
        self.width = weight

    def __call__(self, sample):
        image, gt_map, center_map, weight = sample['image'], sample['gt_map'], \
                                            sample['center_map'], sample['weight']

        image = skimage.transform.resize(image, (self.height, self.width), preserve_range=True)
        return {'image': image, 'gt_map': sample['gt_map'],
                'center_map': sample['center_map'], 'weight': sample['weight']}
