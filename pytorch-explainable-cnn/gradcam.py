import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import utils


class GradCam:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.fmap_pool = {}
        self.grad_pool = {}

        def forward_hook(key):
            def forward_hook_(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        for name, module in self.model.named_modules():
            module.register_forward_hook(forward_hook(name))
            module.register_backward_hook(backward_hook(name))

    def forward(self, inputs):
        inputs = inputs.to(self.device)

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

    def generate(self, target_layer):
        fmaps = self.fmap_pool[target_layer]
        grads = self.grad_pool[target_layer]
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.inputs.shape[2:], mode="bilinear", align_corners=False
        )

        return gcam


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
    parser.add_argument("--target",
                        help="The layer to get the activation map")
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="GPU ID to use.")
    args = parser.parse_args()
    # fmt: on

    return args


def blend_activation_map(x, img, alpha=0.5, cmap_name="jet"):
    # Convert a Tensor to a ndarray.
    x = x.cpu().numpy()
    # Transpose axis from (C, H, W) to (H, W, C)
    x = x.transpose(1, 2, 0)
    # Convert a range of values to [0, 1].
    x = (x - x.min()) / (x.max() - x.min())
    # Convert shape from (H, W, 1) to (H, W).
    x = x.squeeze(2)

    # ヒートマップを作成する。
    cmap = plt.get_cmap(cmap_name)
    heatmap = Image.fromarray(cmap(x, bytes=True)[..., :3])

    # アルファブレンディング
    blended = Image.blend(img, heatmap, alpha=alpha)

    return heatmap, blended


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

    gradcam = GradCam(model)

    results = defaultdict(list)
    for inputs, paths in dataloader:
        # 順伝搬を行う。
        class_ids = gradcam.forward(inputs)

        for i in range(3):  # top 1 ~ top 3
            # i 番目にスコアが高いラベルを取得する。
            ith_class_ids = class_ids[:, i]

            # 逆伝搬を行う。
            gradcam.backward(ith_class_ids)

            # 対象の層の活性化マップを取得する。
            activation_maps = gradcam.generate(args.target)

            for activation_map, path, class_id in zip(
                activation_maps, paths, ith_class_ids
            ):
                img = Image.open(path).convert("RGB")
                # 活性化マップを画像に重ね合わせる。
                heatmap, blended_img = blend_activation_map(activation_map, img)

                result = {
                    "image": img,
                    "blended_img": blended_img,
                    "heatmap": heatmap,
                    "class_name": class_names[class_id],
                }

                results[path].append(result)

    # 結果を保存する。
    utils.output_gradcam_results(args.output, results)


if __name__ == "__main__":
    main()
