import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="epoch number")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
parser.add_argument("--trainsize", type=int, default=384, help="training dataset size")
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
parser.add_argument(
    "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
)
parser.add_argument(
    "--decay_epoch", type=int, default=100, help="every n epochs decay learning rate"
)
parser.add_argument(
    "--load",
    type=str,
    default="bankbone_pth/smt_tiny.pth",
    help="train from checkpoints",
)
parser.add_argument(
    "--load_adm",
    type=str,
    default="bankbone_pth/ADMNet_Plus.pth",
    help="train from checkpoints",
)
parser.add_argument("--gpu_id", type=str, default="0", help="train use gpu")
parser.add_argument(
    "--rgb_root",
    type=str,
    default="dataset/TRAIN/Imgs/",
    help="the training rgb images root",
)
parser.add_argument(
    "--gt_root",
    type=str,
    default="dataset/TRAIN/GT/",
    help="the training gt images root",
)
parser.add_argument(
    "--test_rgb_root",
    type=str,
    default="dataset/TEST/COD10K/Imgs/",
    help="the test gt images root",
)
parser.add_argument(
    "--test_gt_root",
    type=str,
    default="dataset/TEST/COD10K/GT/",
    help="the test gt images root",
)
parser.add_argument(
    "--test_edge_root",
    type=str,
    default="dataset/TEST/COD10K/Edge/",
    help="the test edge images root",
)
parser.add_argument(
    "--edge_root",
    type=str,
    default="dataset/TRAIN/Edge/",
    help="the training edge images root",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="cpts/DFENet/",
    help="the path to save models and logs",
)
parser.add_argument(
    "--train_depth_root",
    type=str,
    default="dataset/TRAIN/deeps/",
    help="the training depth images root",
)
parser.add_argument(
    "--test_depth_root",
    type=str,
    default="dataset/TEST/COD10K/deeps/",
    help="the test depth images root",
)
opt = parser.parse_args()
