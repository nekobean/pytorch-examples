import argparse
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import trange

import yolov3.utils.utils as utils
from yolov3.datasets.coco import COCODataset
from yolov3.models.yolov3 import YOLOv3
from yolov3.utils.parse_yolo_weights import parse_yolo_weights


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="directory path to coco dataset")
    parser.add_argument("--anno_path", type=Path, required=True,
                        help="json filename")
    parser.add_argument("--weights_path", type=Path,
                        help="path to weights file")
    parser.add_argument("--checkpoint_path", type=Path,
                        help="pytorch checkpoint file path")
    parser.add_argument("--config_path", type=Path, default="config/yolov3_coco.yaml",
                        help="path to config file")
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="GPU id to use")
    parser.add_argument("--save_dir", type=Path, default="train_output",
                        help="directory where checkpoint files are saved")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="interval between saving checkpoints")
    # fmt: on
    args = parser.parse_args()

    return args


def build_optimizer(config, model):
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision  # 1反復あたりのサンプル数
    momentum = config["train"]["momentum"]
    decay = config["train"]["decay"]
    base_lr = config["train"]["lr"] / n_samples_per_iter

    params = []
    for key, value in model.named_parameters():
        weight_decay = decay * n_samples_per_iter if "conv.weight" in key else 0
        params.append({"params": value, "weight_decay": weight_decay})

    optimizer = optim.SGD(
        params,
        lr=base_lr,
        momentum=momentum,
        dampening=0,
        weight_decay=decay * n_samples_per_iter,
    )

    return optimizer


def build_scheduler(config, optimizer):
    burn_in = config["train"]["burn_in"]
    steps = config["train"]["steps"]

    def schedule(i):
        if i < burn_in:
            factor = (i / burn_in) ** 4
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01

        return factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, schedule)

    return scheduler


def repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config_path)
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision
    max_iter = config["train"]["max_iter"]
    img_size = config["train"]["img_size"]
    random_resize = config["augmentation"]["random_size"]

    # デバイスを作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # モデルを作成する。
    model = YOLOv3(config["model"], config["train"]["ignore_threshold"])
    if args.weights_path:
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint_path:
        state = torch.load(args.checkpoint_path)
        # モデルの重みを読み込み。
        model.load_state_dict(state["model_state_dict"])
    model = model.to(device).train()

    # オプティマイザを作成する。
    optimizer = build_optimizer(config, model)
    iter_state = 1

    # スケジューラーを作成する。
    scheduler = build_scheduler(config, optimizer)

    if args.checkpoint_path:
        state = torch.load(args.checkpoint_path)
        # オプティマイザの状態を読み込み
        optimizer.load_state_dict(state["optimizer_state_dict"])
        iter_state = state["iter"] + 1

    # Dataset を作成する。
    dataset = COCODataset(
        args.dataset_dir,
        args.anno_path,
        img_size=img_size,
        augmentation=config["augmentation"],
    )

    # DataLoader を作成する。
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    # チェックポイントを保存するディレクトリを作成する。
    args.save_dir.mkdir(exist_ok=True)

    dataloader = repeater(dataloader)
    history = []
    for iter_i in trange(iter_state, max_iter + 1):
        optimizer.zero_grad()
        for _ in range(subdivision):
            imgs, targets, _, _ = next(dataloader)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 順伝搬する。
            loss = model(imgs, targets)

            # 逆伝搬する。
            loss.backward()

        optimizer.step()
        scheduler.step()

        # 学習過程を記録する。
        info = {
            "iter": iter_i,
            "lr": scheduler.get_lr()[0] * n_samples_per_iter,
            "loss_xy": float(model.loss_dict["xy"]),
            "loss_wh": float(model.loss_dict["wh"]),
            "loss_obj_conf": float(model.loss_dict["conf"]),
            "loss_class_conf": float(model.loss_dict["cls"]),
            "loss_total": float(model.loss_dict["l2"]),
            "img_size": img_size,
        }
        history.append(info)

        # モデルの入力サイズを変更する。
        if random_resize and iter_i % 10 == 0:
            img_size = np.random.randint(10, 20) * 32
            dataset.img_size = img_size
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
            )
            dataloader = repeater(dataloader)

        # チェックポイントを保存する。
        if iter_i % args.save_interval == 0 or iter_i == max_iter:
            # モデルを保存する。
            save_path = args.save_dir / f"yolov3_{iter_i}.pth"
            state_dict = {
                "iter": iter_i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state_dict, save_path)

            # 学習経過を保存する。
            save_path = args.save_dir / "history.csv"
            pd.DataFrame(history).to_csv(save_path, index=False)


if __name__ == "__main__":
    main()