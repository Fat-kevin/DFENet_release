import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


def cv_random_flip(img, label, edge, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge, depth


def randomCrop(image, label, edge, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return (
        image.crop(random_region),
        label.crop(random_region),
        edge.crop(random_region),
        depth.crop(random_region),
    )


def randomRotation(image, label, edge, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, edge, depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, train_depth_root, trainsize):
        self.trainsize = trainsize
        self.images = [
            image_root + f for f in os.listdir(image_root) if f.endswith(".jpg")
        ]
        self.gts = [
            gt_root + f
            for f in os.listdir(gt_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.edges = [
            edge_root + f
            for f in os.listdir(edge_root)
            if f.endswith(".bmp") or f.endswith(".png") or f.endswith(".jpg")
        ]
        self.depths = [
            train_depth_root + f
            for f in os.listdir(train_depth_root)
            if f.endswith(".bmp") or f.endswith(".png") or f.endswith(".jpg")
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform_15 = transforms.Compose(
            [
                transforms.Resize(
                    (int(self.trainsize * 1.5), int(self.trainsize * 1.5))
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )
        self.edges_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()]
        )

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])
        depth = self.rgb_loader(self.depths[index])
        image, gt, edge, depth = cv_random_flip(image, gt, edge, depth)
        image, gt, edge, depth = randomCrop(image, gt, edge, depth)
        image, gt, edge, depth = randomRotation(image, gt, edge, depth)
        image_raw = colorEnhance(image)
        gt_raw = randomPeper(gt)
        image15 = self.img_transform_15(image_raw)
        gt = self.gt_transform(gt_raw)
        edge = self.edges_transform(edge)
        depth = self.depths_transform(depth)
        return image15, gt, edge, depth

    def filter_files(self):
        assert (
            len(self.images) == len(self.gts) == len(self.edges) == len(self.depths)
        ), "images, gts, edges, and depths must have the same number of files"
        images = []
        gts = []
        edges = []
        depths = []
        for img_path, gt_path, edge_path, depth_path in zip(
            self.images, self.gts, self.edges, self.depths
        ):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                edge = Image.open(edge_path)
                depth = Image.open(depth_path)
                if img.size == gt.size == edge.size == depth.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    edges.append(edge_path)
                    depths.append(depth_path)
                else:
                    print(
                        f"Size mismatch: {img_path} ({img.size}), {gt_path} ({gt.size}), {edge_path} ({edge.size})"
                    )
            except Exception as e:
                print(
                    f"Error processing files: {img_path}, {gt_path}, {edge_path}. Error: {e}"
                )
        self.images = images
        self.gts = gts
        self.edges = edges
        self.depths = depths
        print(
            f"Filtered data: {len(self.images)} images, {len(self.gts)} GTs, {len(self.edges)} edges ,{len(self.depths)} deeps"
        )

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    gt_root,
    edge_root,
    train_depth_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
):
    dataset = SalObjDataset(image_root, gt_root, edge_root, train_depth_root, trainsize)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, test_depth_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [
            gt_root + f
            for f in os.listdir(gt_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.depths = [
            test_depth_root + f
            for f in os.listdir(test_depth_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform_15 = transforms.Compose(
            [
                transforms.Resize((int(self.testsize * 1.5), int(self.testsize * 1.5))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()]
        )
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_15 = self.transform_15(image).unsqueeze(0)
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.gts[self.index].split("/")[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith(".jpg"):
            name = name.split(".jpg")[0] + ".jpg"
        self.index += 1
        self.index = self.index % self.size
        return image_15, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):
        return self.size
