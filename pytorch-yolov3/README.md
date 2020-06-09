# YOLOv3

## 重みファイルのダウンロード

```bash
wget -P weights https://pjreddie.com/media/files/yolov3.weights
wget -P weights https://pjreddie.com/media/files/darknet53.conv.74
```

## 1枚の画像に対して、推論を行う。

```bash
python detect_image.py \
    --input_path data/dog.png \
    --weights_path weights/yolov3.weights \
    --gpu_id 0
```

## ディレクトリ内の画像に対して、推論を行う。

```bash
python detect_image.py \
    --input_path data \
    --weights_path weights/yolov3.weights \
    --gpu_id 0
```

## COCO データセットで mAP を計算する

```bash
python evaluate_coco.py \
    --dataset_dir /data/COCO \
    --anno_path config/instances_val5k.json \
    --weights_path weights/yolov3.weights \
    --gpu_id 0
```

```txt
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.313
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.339
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.457
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.416
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.603
0.31082299897937926 0.5579624636166972
```

## COCO データセットで学習する

```bash
python train_coco.py \
    --dataset_dir /data/COCO \
    --anno_path /data/COCO/annotations/instances_train2017.json \
    --weights_path weights/darknet53.conv.74 \
    --gpu_id 0
```
