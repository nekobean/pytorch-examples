import json
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
from PIL import Image


# Ensure Reproducibility.
torch.backends.cudnn.deterministic = True


def check_array(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()

    b = np.load(b)
    assert a.shape == b.shape
    # assert a.dtype == b.dtype

    if np.issubdtype(a.dtype, np.integer):
        equals = (np.abs(a.astype(int) - b.astype(int)).max() <= 1).all()
    else:
        equals = np.allclose(a, b, rtol=1e-03, atol=1e-03)

    if not equals:
        print(f"a: {a.shape} {a.dtype}")
        print(f"b: {b.shape} {b.dtype}")
        print(f"a: {a.ravel()[:10]}")
        print(f"b: {b.ravel()[:10]}")
        assert False


class ImageFolder(data.Dataset):
    """Dataset for loading an image from a specified directory.
    """

    def __init__(self, img_dir, transform=None):
        self.paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, str(path)

    def __len__(self):
        return len(self.paths)

    def _get_img_paths(self, img_dir):
        """Get a list of image paths in a specified directory.

        Args:
            img_dir (Path): Directory to load images.

        Returns:
            list of Paths: List of image paths in the specified directory.
        """
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        paths = sorted(x for x in img_dir.iterdir() if x.suffix in extensions)

        return paths


def get_classes():
    """Get a list of ImageNet classes.

    Returns:
        list of str: A list of ImageNet classes.
    """
    if not Path("data/imagenet_class_index.json").exists():
        datasets.utils.download_url(
            "https://git.io/JebAs", "data", "imagenet_class_index.json"
        )

    with open("data/imagenet_class_index.json") as f:
        json_dict = json.load(f)
        class_names = [x["ja"] for x in json_dict]

    return class_names


def tensor_to_image(x):
    """Convert a Tensor to a image.

    Args:
        x (Tensor): (C, H, W) Tensor.

    Returns:
        ndarray: (H, W, C) ndarray.
    """
    # Convert a Tensor to a ndarray.
    x = x.cpu().numpy()
    # Transpose axis from (C, H, W) to (H, W, C)
    x = x.transpose(1, 2, 0)
    # Convert a range of values to [0, 1].
    x = (x - x.min()) / (x.max() - x.min())
    # Convert a range of values from [0, 1] to [0, 255].
    x = np.uint8(x * 255)

    if x.shape[2] == 1:
        # Convert shape from (H, W, 1) to (H, W).
        x = x.squeeze(2)

    return x


def tensor_to_gray_image(x):
    """Convert a Tensor to a grayscale image.

    Args:
        x (Tensor): (C, H, W) Tensor.

    Returns:
        ndarray: (H, W) ndarray.
    """
    # Calculate the sum of all channels. (3, H, W) -> (1, H, W)
    x = x.abs().sum(dim=0, keepdims=True)
    x = tensor_to_image(x)

    return x


def tensor_to_positive_saliency(x):
    """Convert a Tensor to a grayscale image clipped to negative values to 0.

    Args:
        x (Tensor): (C, H, W) Tensor.

    Returns:
        ndarray: (H, W) ndarray.
    """
    # clipped to negative values to 0.
    x = torch.nn.functional.relu(x)
    x = tensor_to_image(x)

    return x


def tensor_to_negative_saliency(x):
    """Convert a Tensor to a grayscale image clipped to positive values to 0.

    Args:
        x (Tensor): (C, H, W) Tensor.

    Returns:
        ndarray: (H, W) ndarray.
    """
    # clipped to positive values to 0.
    x = torch.nn.functional.relu(-x)
    x = tensor_to_image(x)

    return x


def to_onehot(y, n_classes):
    """Convert a list of labels to onehot representation.

    Args:
        y (list of int): A list of labels.
        n_classes (int): Number of classes.

    Returns:
        (N, NumClasses) Tensor: onehot representation
    """
    # (N,) tensor to (N, NumClasses) tensor.
    return torch.eye(n_classes)[y]


def get_device(gpu_id=-1):
    """Get a device to use.

    Args:
        use_gpu (int): GPU ID to use. When using the CPU, specify -1.

    Returns:
        torch.device: Device to use.
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")


def create_model(name, pretrained=True):
    """Create a model with a given name.

    Args:
        name (str): Model name to create.
        pretrained (bool): Whether to load pretrained weights.

    Raises:
        ValueError: Raised if the model with the specified name does not exist in torchvision.

    Returns:
        torch.nn.Module: model
    """
    # Create a list of available models.
    available_models = {}
    for obj_name in dir(torchvision.models):
        obj = getattr(torchvision.models, obj_name)
        if inspect.isfunction(obj):
            available_models[obj_name] = obj

    # Create a model with a given name.
    if name not in available_models:
        raise ValueError(f"The model {name} does not exist in torchvision.")

    model = available_models[name](pretrained=pretrained)

    return model


def output_results(output_dir, results, name):
    """Save the results as an image.

    Args:
        output_dir (Path): The directory where the image is saved.
        results (dict): Result dict.
        name (str): File name to be saved.
    """
    output_dir.mkdir(exist_ok=True)

    for path, result in results.items():
        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(4, 4, 1)
        ax.set_axis_off()
        ax.imshow(result[0]["image"])

        for i in range(3):
            color_img = result[i]["color_grad"]
            gray_img = result[i]["gray_grad"]
            pos_grad = result[i]["pos_grad"]
            neg_grad = result[i]["neg_grad"]
            class_name = result[i]["class_name"]

            ax = fig.add_subplot(4, 4, i * 4 + 5)
            ax.imshow(color_img)
            ax.set_ylabel(f"Top {i + 1} - {class_name}")
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(4, 4, i * 4 + 6)
            ax.imshow(gray_img, cmap="jet")
            ax.set_axis_off()

            ax = fig.add_subplot(4, 4, i * 4 + 7)
            ax.imshow(pos_grad, cmap="jet")
            ax.set_axis_off()

            ax = fig.add_subplot(4, 4, i * 4 + 8)
            ax.imshow(neg_grad, cmap="jet")
            ax.set_axis_off()

        axes = fig.get_axes()
        axes[1].set_title("Color Gradient")
        axes[2].set_title("Grayscale Gradient")
        axes[3].set_title("Positive Gradient")
        axes[4].set_title("Negative Gradient")

        save_path = output_dir / f"{name}_{Path(path).stem}.png"
        fig.savefig(save_path, bbox_inches="tight")


def output_gradcam_results(output_dir, results):
    """Save the results as an image.

    Args:
        output_dir (Path): The directory where the image is saved.
        results (dict): Result dict.
    """
    output_dir.mkdir(exist_ok=True)

    for path, result in results.items():
        fig = plt.figure(figsize=(7, 10))

        ax = fig.add_subplot(4, 3, 1)
        ax.set_axis_off()
        ax.imshow(result[0]["image"])

        for i in range(3):
            heatmap = result[i]["heatmap"]
            blended_img = result[i]["blended_img"]
            class_name = result[i]["class_name"]

            ax = fig.add_subplot(4, 3, i * 3 + 4)
            ax.imshow(heatmap)
            ax.set_ylabel(f"Top {i + 1} - {class_name}")
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(4, 3, i * 3 + 5)
            ax.imshow(blended_img, cmap="jet")
            ax.set_axis_off()

        axes = fig.get_axes()
        axes[1].set_title("Activation Map")
        axes[2].set_title("Blend Image")

        save_path = output_dir / f"gradcam_{Path(path).stem}.png"
        fig.savefig(save_path, bbox_inches="tight")
