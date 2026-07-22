import os
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("./models")
import numpy as np
from datetime import datetime
from models.DFENet import DFENet
from torchvision.utils import make_grid
from data_cod import get_loader, test_dataset
from utils import clip_gradient
import logging
import random
import torch.backends.cudnn as cudnn
from options_cod import opt
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

seed = None
seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_warmup_cosine_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


if opt.gpu_id == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("USE GPU 0")
elif opt.gpu_id == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("USE GPU 1")
cudnn.benchmark = False
image_root = opt.rgb_root
gt_root = opt.gt_root
edge_root = opt.edge_root
train_depth_root = opt.train_depth_root
test_depth_root = opt.test_depth_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path
log_dir = os.path.dirname(save_path + "DFENet.log")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=save_path + "DFENet.log",
    format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
    level=logging.INFO,
    filemode="a",
    datefmt="%Y-%m-%d %I:%M:%S %p",
)
logging.info("DFENet-Train_4_pairs")
logging.info("Random Seed: {}".format(seed))
logging.info("Learning Rate: {}".format(opt.lr))
logging.info("Batch Size: {}".format(opt.batchsize))
logging.info("Epochs: {}".format(opt.epoch))
model = DFENet()
num_parms = 0
if opt.load is not None:
    model.load_pre(opt.load)
    print("load model from ", opt.load)
    model.load_pre2(opt.load)
    print("load model2 from ", opt.load)
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))
params = model.parameters()
optimizer = torch.optim.AdamW(params, lr=opt.lr)
warmup_steps = 30
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=180, T_mult=1, eta_min=opt.lr * 0.01
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)
if not os.path.exists(save_path):
    os.makedirs(save_path)
print("load data...")
train_loader = get_loader(
    image_root,
    gt_root,
    edge_root,
    train_depth_root,
    batchsize=opt.batchsize,
    trainsize=opt.trainsize,
)
test_loader = test_dataset(
    test_image_root, test_gt_root, test_depth_root, opt.trainsize
)
total_step = len(train_loader)
logging.info("Config")
logging.info(
    "epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};load:{};save_path:{}".format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.load, save_path
    )
)
CE = torch.nn.BCEWithLogitsLoss()
step = 0
writer = SummaryWriter(save_path + "summary")
best_mae = 1
best_epoch = 0


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda()
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        tq = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            ncols=80,
            desc=f"[Train {epoch:03d}]",
        )
        for i, (image_m, gt_m, gt_edge, depth) in tq:
            optimizer.zero_grad()
            image_m = image_m.cuda()
            gt_m = gt_m.cuda()
            gt_edge = gt_edge.cuda()
            depth = depth.cuda()
            s1, s2, s3, s4, edge_map = model(image_m, depth)
            bce_iou1 = CE(s1, gt_m) + iou_loss(s1, gt_m)
            bce_iou2 = CE(s2, gt_m) + iou_loss(s2, gt_m)
            bce_iou3 = CE(s3, gt_m) + iou_loss(s3, gt_m)
            bce_iou4 = CE(s4, gt_m) + iou_loss(s4, gt_m)
            edge_loss = dice_loss(edge_map, gt_edge)
            loss = bce_iou1 + bce_iou2 + bce_iou3 + bce_iou4 + 3 * edge_loss
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.item()
            tq.set_postfix(loss=f"{loss.item():.4f}")
            if i % 100 == 0 or i == len(train_loader) or i == 1:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || loss:{:4f}".format(
                        datetime.now(),
                        epoch,
                        opt.epoch,
                        i,
                        len(train_loader),
                        current_lr,
                        loss.item(),
                    )
                )
                logging.info(
                    "TRAIN: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss:{:4f}".format(
                        epoch, opt.epoch, i, len(train_loader), current_lr, loss.item()
                    )
                )
                writer.add_scalar("Loss", loss.item(), global_step=step)
                grid_image = make_grid(depth[0].clone().cpu().data, 1, normalize=True)
                writer.add_image("depth", grid_image, step)
                grid_image = make_grid(image_m[0].clone().cpu().data, 1, normalize=True)
                writer.add_image("RGB", grid_image, step)
                grid_image = make_grid(gt_m[0].clone().cpu().data, 1, normalize=True)
                writer.add_image("Ground_truth", grid_image, step)
                grid_edge_image = make_grid(
                    gt_edge[0].clone().cpu().data, 1, normalize=True
                )
                writer.add_image("Ground_truth_Edge", grid_edge_image, step)
                res = s1[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image("res", torch.tensor(res), step, dataformats="HW")
                edge_map_res = (
                    edge_map[0].clone().sigmoid().data.cpu().numpy().squeeze()
                )
                edge_map_res = (edge_map_res - edge_map_res.min()) / (
                    edge_map_res.max() - edge_map_res.min() + 1e-8
                )
                writer.add_image(
                    "Edge_map", torch.tensor(edge_map_res), step, dataformats="HW"
                )
        loss_all /= epoch_step
        logging.info(
            "TRAIN: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}".format(
                epoch, opt.epoch, loss_all
            )
        )
        writer.add_scalar("Loss-epoch", loss_all, global_step=epoch)
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), save_path + "DFENet_epoch_{}.pth".format(epoch)
            )
    except KeyboardInterrupt:
        print("Keyboard Interrupt: save model and exit.")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(
            model.state_dict(), save_path + "DFENet_epoch_{}.pth".format(epoch + 1)
        )
        print("save checkpoints successfully!")
        raise


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    mae_sum = 0
    tq = tqdm(range(test_loader.size), ncols=80, desc=f"[Test  {epoch:03d}]")
    with torch.no_grad():
        for i in tq:
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            image = image.cuda()
            depth = depth.cuda()
            res, res2, res3, res4, edge, d1, d3 = model(image, depth)
            res = F.interpolate(
                res, size=gt.shape, mode="bilinear", align_corners=False
            )
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae = np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])
            mae_sum += mae
            tq.set_postfix(mae=f"{mae:.4f}")
        mae = mae_sum / test_loader.size
        writer.add_scalar("MAE", torch.tensor(mae), global_step=epoch)
        print(
            "Epoch: {} MAE: {} bestMAE: {} bestEpoch: {}".format(
                epoch, mae, best_mae, best_epoch
            )
        )
        if epoch == 1 or mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + "DFENet_epoch_best.pth")
            print("best epoch:{}".format(epoch))
        logging.info(
            "TEST: Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}".format(
                epoch, mae, best_epoch, best_mae
            )
        )


VALIDATION_POLICY = {
    "first_epoch": True,
    "skip_until": 120,
    "interval_start": 121,
    "interval_end": 120,
    "interval_step": 1,
}


def should_validate(epoch):
    if epoch == 1 and VALIDATION_POLICY["first_epoch"]:
        return True
    if epoch <= VALIDATION_POLICY["skip_until"]:
        return False
    if (
        VALIDATION_POLICY["interval_start"]
        <= epoch
        <= VALIDATION_POLICY["interval_end"]
    ):
        return epoch % VALIDATION_POLICY["interval_step"] == 0
    if epoch > VALIDATION_POLICY["interval_end"]:
        return True
    return False
