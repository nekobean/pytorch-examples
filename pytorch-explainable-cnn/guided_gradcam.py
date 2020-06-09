import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import utils
from gradcam import GradCam
from guided_backprop import GuidedBackprop


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


def main():
    args = parse_args()

    # デバイスを選択する。
    device = utils.get_device(args.gpu_id)

    # モデルを作成する。 (hook 関数を使うので、同じ実態のモデルを両方で使えない)
    model_for_gradcam = utils.create_model(args.model).to(device).eval()
    model_for_guided_backprop = utils.create_model(args.model).to(device).eval()

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

    gradcam = GradCam(model_for_gradcam)
    guided_backprop = GuidedBackprop(model_for_guided_backprop)

    results = defaultdict(list)
    for inputs, paths in dataloader:
        # 順伝搬を行う。
        class_ids = gradcam.forward(inputs)
        class_ids = guided_backprop.forward(inputs)

        for i in range(3):  # top 1 ~ top 3
            # i 番目にスコアが高いラベルを取得する。
            ith_class_ids = class_ids[:, i]

            # 逆伝搬を行う。
            gradcam.backward(ith_class_ids)
            guided_backprop.backward(ith_class_ids)

            # 対象の層の活性化マップを取得する。
            regions = gradcam.generate(args.target)

            # 入力画像の勾配を取得する。
            gradients = guided_backprop.generate()

            for grad, region, path, class_id in zip(
                gradients, regions, paths, ith_class_ids
            ):
                guided_cam = torch.mul(region, grad)
                result = {
                    "image": Image.open(path).convert("RGB"),
                    "color_grad": utils.tensor_to_image(guided_cam),
                    "gray_grad": utils.tensor_to_gray_image(grad),
                    "pos_grad": utils.tensor_to_positive_saliency(grad),
                    "neg_grad": utils.tensor_to_negative_saliency(grad),
                    "class_name": class_names[class_id],
                }

                results[path].append(result)

    # 結果を保存する。
    utils.output_results(args.output, results, "guided_gradcam")


if __name__ == "__main__":
    main()
