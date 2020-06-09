import argparse
from collections import defaultdict
from pathlib import Path

import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import utils


class VanillaBackprop:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, inputs):
        inputs = inputs.to(self.device).requires_grad_()

        # 勾配を初期化する。
        self.model.zero_grad()

        # 順伝搬を行う。
        logits = self.model(inputs)

        # softmax を適用する。
        probs = F.softmax(logits, dim=1)

        # 確率が大きい順にソートする。
        class_ids = probs.sort(dim=1, descending=True)[1]

        self.inputs = inputs
        self.logits = logits

        return class_ids

    def backward(self, class_ids):
        # onehot 表現に変換する。
        onehot = utils.to_onehot(class_ids, n_classes=self.logits.size(1))
        onehot = onehot.to(self.device)

        # 逆伝搬を行う。
        self.logits.backward(gradient=onehot, retain_graph=True)

    def generate(self):
        # 入力層の勾配を取得する。
        gradients = self.inputs.grad.clone()
        # 入力層の勾配を初期化する。
        self.inputs.grad.zero_()

        return gradients


def parse_args():
    """Parse command line arguments.
    """
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        help="Model name to use.")
    parser.add_argument("--input", type=Path,
                        help="Path of the directory containing the image to infer.")
    parser.add_argument("--output", type=Path, default="output",
                        help="Path of the directory to output the results.")
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="GPU ID to use.")
    args = parser.parse_args()
    # fmt: on

    return args


def main():
    args = parse_args()

    # デバイスを選択する。
    device = utils.get_device(args.gpu_id)

    # モデルを作成する。
    model = utils.create_model(args.model).to(device).eval()

    # Transform を作成する。
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset を作成する。
    dataset = utils.ImageFolder(args.input, transform)

    # DataLoader を作成する。
    dataloader = data.DataLoader(dataset, batch_size=4)

    # クラス名一覧を取得する。
    class_names = utils.get_classes()

    vanilla_backprop = VanillaBackprop(model)

    results = defaultdict(list)
    for inputs, paths in dataloader:
        # 順伝搬を行う。
        class_ids = vanilla_backprop.forward(inputs)

        for i in range(3):  # top 1 ~ top 3
            # i 番目にスコアが高いラベルを取得する。
            ith_class_ids = class_ids[:, i]

            # 逆伝搬を行う。
            vanilla_backprop.backward(ith_class_ids)

            # 入力画像に対する勾配を取得する。
            gradients = vanilla_backprop.generate()

            for grad, path, class_id in zip(gradients, paths, ith_class_ids):
                result = {
                    "image": Image.open(path).convert("RGB"),
                    "color_grad": utils.tensor_to_image(grad),
                    "gray_grad": utils.tensor_to_gray_image(grad),
                    "pos_grad": utils.tensor_to_positive_saliency(grad),
                    "neg_grad": utils.tensor_to_negative_saliency(grad),
                    "class_name": class_names[class_id],
                }

                results[path].append(result)

    # 結果を保存する。
    utils.output_results(args.output, results, "vanilla_backprop")


if __name__ == "__main__":
    main()
