import cv2
import torch
import torchvision.transforms as transforms

import yolov3.utils.utils as utils


class ImageFolder(torch.utils.data.Dataset):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, img_path_or_dir, img_size):
        super().__init__()
        if img_path_or_dir.is_dir():
            self.img_paths = self.get_img_paths(img_path_or_dir)
        else:
            self.img_paths = [img_path_or_dir]
        self.img_size = img_size

    def __getitem__(self, index):
        path = str(self.img_paths[index])
        # 画像を読み込む。
        img = cv2.imread(path)
        # チャンネルの順番を変更する。 (B, G, R) -> (R, G, B)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # レターボックス化する。
        img, pad_info = utils.letterbox(img, self.img_size)
        # PIL -> Tensor
        img = transforms.ToTensor()(img)

        return img, pad_info, path

    def __len__(self):
        return len(self.img_paths)

    def get_img_paths(self, img_dir):
        img_paths = [
            x for x in img_dir.iterdir() if x.suffix in ImageFolder.IMG_EXTENSIONS
        ]
        img_paths.sort()

        return img_paths