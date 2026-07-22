import time
import torch
import torch.nn.functional as F
import sys

sys.path.append("./models")
import numpy as np
import os, argparse
import cv2
from models.DFENet import DFENet
from data_cod import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=384, help="testing size")
parser.add_argument("--gpu_id", type=str, default="0", help="select gpu id")
parser.add_argument(
    "--test_path", type=str, default="dataset/TEST/", help="test dataset path"
)
opt = parser.parse_args()
dataset_path = opt.test_path
if opt.gpu_id == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("USE GPU 0")
elif opt.gpu_id == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("USE GPU 1")
model = DFENet()
model_path = "cpts/DFENet/DFENet_epoch_best.pth"
saved_weights = torch.load(model_path)
model.load_state_dict(saved_weights, strict=False)
model.cuda()
model.eval()
test_datasets = ["CAMO", "CHAMELEON", "COD10K", "NC4K"]
for dataset in test_datasets:
    save_path = "./test_maps/DFENet/" + dataset + "/"
    edge_save_path = save_path + "edge_maps/"
    depth_path = "./dataset/TEST/" + dataset + "/deeps/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + "/Imgs/"
    gt_root = dataset_path + dataset + "/GT/"
    test_loader = test_dataset(image_root, gt_root, depth_path, opt.testsize)
    total_time = 0
    count = 0
    for i in range(test_loader.size):
        image, gt, depth, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= gt.max() + 1e-8
        image = image.cuda()
        depth = depth.cuda()
        start_time = time.perf_counter()
        res, res2, res3, res4, edge, d1, d3 = model(image, depth)
        end_time = time.perf_counter()
        count += 1
        total_time += end_time - start_time
        res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
        edge_resized = F.upsample(
            edge, size=gt.shape, mode="bilinear", align_corners=False
        )
        edge_resized = edge_resized.sigmoid().data.cpu().numpy().squeeze()
        edge_resized = (edge_resized - edge_resized.min()) / (
            edge_resized.max() - edge_resized.min() + 1e-8
        )
        cv2.imwrite(edge_save_path + name, edge_resized * 255)
    fps = count / total_time
    print(f"FPS for {dataset}:", fps)
print("Test Done!")
