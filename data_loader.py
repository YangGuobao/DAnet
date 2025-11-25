
# coding: utf-8
# @Author : 杨国宝
# @Time : 2025/5/12 9:57

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance


def cv_random_flip(img_A, img_B, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label

def randomRotation(image_A, image_B, label):
    # 标签旋转使用NEAREST以保持类别值
    mode_img = Image.BICUBIC
    mode_label = Image.NEAREST
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image_A = image_A.rotate(random_angle, mode_img)
        image_B = image_B.rotate(random_angle, mode_img)
        label = label.rotate(random_angle, mode_label)
    return image_A, image_B, label

def colorEnhance(image_A, image_B):
    images_to_enhance = [image_A, image_B]
    enhanced_images = []
    for image in images_to_enhance:
        bright_intensity = random.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.uniform(0.7, 1.3)
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.uniform(0.7, 1.3)
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        enhanced_images.append(image)
    return enhanced_images[0], enhanced_images[1]

# --- 训练数据集类 ---
class ChangeDataset(data.Dataset):
    def __init__(self, root, trainsize):
        self.trainsize = trainsize
        self.image_root_A = os.path.join(root, 'A/')
        self.image_root_B = os.path.join(root, 'B/')
        self.gt_root = os.path.join(root, 'label/')

        # 获取A目录下的所有图像文件名 (常用格式)
        self.images_A_files = sorted([f for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))])

        # 构建完整路径
        self.images_A = [os.path.join(self.image_root_A, f) for f in self.images_A_files]
        self.images_B = [os.path.join(self.image_root_B, f) for f in self.images_A_files] # 假设文件名相同
        self.gts = [os.path.join(self.gt_root, f) for f in self.images_A_files] # 假设文件名相同

        self.filter_files() # 过滤掉不存在或尺寸不匹配的文件

        # 定义图像和标签的转换
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.size = len(self.images_A)
        print(f"训练集初始化完成，找到 {self.size} 个有效样本。")

    def __getitem__(self, index):
        image_A, image_B, gt = self.load_img_and_mask(index)

        # 应用数据增强
        image_A, image_B, gt = cv_random_flip(image_A, image_B, gt)
        image_A, image_B, gt = randomRotation(image_A, image_B, gt)
        image_A, image_B = colorEnhance(image_A, image_B)

        # 应用转换
        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)

        # 确保标签是二值的 (0或1) 且为 float 类型
        gt = (gt > 0.5).float() # 假设原始标签像素值在[0, 255]范围，ToTensor会归一化到[0, 1]

        return image_A, image_B, gt

    def load_img_and_mask(self, index):
        try:
            A = Image.open(self.images_A[index]).convert('RGB')
            B = Image.open(self.images_B[index]).convert('RGB')
            mask = Image.open(self.gts[index]).convert('L') # 标签读为灰度图
            return A, B, mask
        except Exception as e:
            print(f"Error loading index {index}: {e}")
            # 处理错误，例如返回 None 或重新尝试加载其他索引
            # 为了简单起见，这里重新加载第一个样本
            A = Image.open(self.images_A[0]).convert('RGB')
            B = Image.open(self.images_B[0]).convert('RGB')
            mask = Image.open(self.gts[0]).convert('L')
            return A, B, mask

    def filter_files(self):
        valid_indices = []
        original_count = len(self.images_A)
        print(f"开始过滤文件，初始样本数: {original_count}")
        for i in range(original_count):
            img_A_path = self.images_A[i]
            img_B_path = self.images_B[i]
            gt_path = self.gts[i]

            if not (os.path.exists(img_A_path) and os.path.exists(img_B_path) and os.path.exists(gt_path)):
                # print(f"Skipping index {i}: File missing.")
                continue
            # 可选：添加尺寸检查
            # try:
            #     img_A_size = Image.open(img_A_path).size
            #     img_B_size = Image.open(img_B_path).size
            #     gt_size = Image.open(gt_path).size
            #     if img_A_size == img_B_size and img_A_size == gt_size:
            #         valid_indices.append(i)
            #     # else:
            #     #     print(f"Skipping index {i}: Size mismatch.")
            # except Exception as e:
            #     # print(f"Error checking size for index {i}: {e}")
            #     continue
            valid_indices.append(i) # 简化：仅检查文件是否存在

        self.images_A_files = [self.images_A_files[i] for i in valid_indices]
        self.images_A = [self.images_A[i] for i in valid_indices]
        self.images_B = [self.images_B[i] for i in valid_indices]
        self.gts = [self.gts[i] for i in valid_indices]
        print(f"文件过滤完成，有效样本数: {len(self.images_A)}")

    def __len__(self):
        return self.size

# --- 获取训练加载器 ---
def get_loader(root, batchsize, trainsize, num_workers=1, shuffle=True, pin_memory=True):
    dataset = ChangeDataset(root=root, trainsize=trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True) # 训练时通常丢弃最后一个不足 batch 的数据
    return data_loader

# --- 测试数据集类 (增加了返回文件名) ---
class Test_ChangeDataset(data.Dataset):
    def __init__(self, root, testsize):
        self.testsize = testsize
        self.image_root_A = os.path.join(root, 'A/')
        self.image_root_B = os.path.join(root, 'B/')
        self.gt_root = os.path.join(root, 'label/')

        self.images_A_files = sorted([f for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))])
        self.images_A = [os.path.join(self.image_root_A, f) for f in self.images_A_files]
        self.images_B = [os.path.join(self.image_root_B, f) for f in self.images_A_files]
        self.gts = [os.path.join(self.gt_root, f) for f in self.images_A_files]

        self.filter_files() # 过滤

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.size = len(self.images_A)
        print(f"测试/验证集初始化完成，找到 {self.size} 个有效样本。")

    def __getitem__(self, index):
        image_A, image_B, gt = self.load_img_and_mask(index)

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)
        gt = (gt > 0.5).float() # 确保二值化

        file_name = os.path.splitext(self.images_A_files[index])[0] # 获取不带后缀的文件名

        return image_A, image_B, gt, file_name # 返回文件名用于保存结果

    def load_img_and_mask(self, index):
        # 与训练集中的函数相同
        try:
            A = Image.open(self.images_A[index]).convert('RGB')
            B = Image.open(self.images_B[index]).convert('RGB')
            mask = Image.open(self.gts[index]).convert('L')
            return A, B, mask
        except Exception as e:
            print(f"Error loading test index {index}: {e}")
            A = Image.open(self.images_A[0]).convert('RGB')
            B = Image.open(self.images_B[0]).convert('RGB')
            mask = Image.open(self.gts[0]).convert('L')
            return A, B, mask

    def filter_files(self):
        # 与训练集中的函数相同
        valid_indices = []
        original_count = len(self.images_A)
        print(f"开始过滤测试/验证文件，初始样本数: {original_count}")
        for i in range(original_count):
            img_A_path = self.images_A[i]
            img_B_path = self.images_B[i]
            gt_path = self.gts[i]
            if not (os.path.exists(img_A_path) and os.path.exists(img_B_path) and os.path.exists(gt_path)):
                continue
            valid_indices.append(i)
        self.images_A_files = [self.images_A_files[i] for i in valid_indices]
        self.images_A = [self.images_A[i] for i in valid_indices]
        self.images_B = [self.images_B[i] for i in valid_indices]
        self.gts = [self.gts[i] for i in valid_indices]
        print(f"文件过滤完成，有效样本数: {len(self.images_A)}")

    def __len__(self):
        return self.size

# --- 获取测试/验证加载器 ---
def get_test_loader(root, batchsize, testsize, num_workers=1, shuffle=False, pin_memory=True):
    dataset = Test_ChangeDataset(root=root, testsize=testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle, # 测试时通常不打乱
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False) # 测试时通常不丢弃
    return data_loader


# --- (data_loader.py 文件中 get_test_loader 函数之后) ---

# --- 新增：专门用于预测的数据集类 ---
class PredictionDataset(data.Dataset):
    """
    专门用于预测的数据集，只加载 A 和 B 影像，不加载也不检查 label。
    """

    def __init__(self, root, testsize):
        self.testsize = testsize
        self.image_root_A = os.path.join(root, 'A/')
        self.image_root_B = os.path.join(root, 'B/')

        # 仅从 A 目录获取文件列表
        try:
            self.images_A_files = sorted(
                [f for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))])
        except FileNotFoundError:
            print(f"错误：找不到 A 目录: {self.image_root_A}")
            self.images_A_files = []

        self.images_A = [os.path.join(self.image_root_A, f) for f in self.images_A_files]
        self.images_B = [os.path.join(self.image_root_B, f) for f in self.images_A_files]

        self.filter_files()  # 过滤（只检查 A 和 B 是否存在）

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images_A)
        print(f"预测集初始化完成，找到 {self.size} 个有效样本。")

    def __getitem__(self, index):
        image_A = self.rgb_loader(self.images_A[index])
        image_B = self.rgb_loader(self.images_B[index])

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)

        file_name = os.path.splitext(self.images_A_files[index])[0]  # 获取不带后缀的文件名

        return image_A, image_B, file_name  # 只返回 T1, T2 和文件名

    def filter_files(self):
        valid_indices = []
        original_count = len(self.images_A)
        print(f"开始过滤预测文件，初始样本数: {original_count}")
        for i in range(original_count):
            img_A_path = self.images_A[i]
            img_B_path = self.images_B[i]

            # --- 关键修改：只检查 A 和 B ---
            if not (os.path.exists(img_A_path) and os.path.exists(img_B_path)):
                # print(f"Skipping index {i}: File A or B missing.")
                continue

            # 可选：尺寸检查
            try:
                img_A_size = Image.open(img_A_path).size
                img_B_size = Image.open(img_B_path).size
                if img_A_size == img_B_size:
                    valid_indices.append(i)
            except Exception:
                continue
            # --- 修改结束 ---

        self.images_A_files = [self.images_A_files[i] for i in valid_indices]
        self.images_A = [self.images_A[i] for i in valid_indices]
        self.images_B = [self.images_B[i] for i in valid_indices]
        print(f"文件过滤完成，有效样本数: {len(self.images_A)}")

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size


# --- 新增：获取预测加载器 ---
def get_predict_loader(root, batchsize, testsize, num_workers=1, shuffle=False, pin_memory=True):
    dataset = PredictionDataset(root=root, testsize=testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader