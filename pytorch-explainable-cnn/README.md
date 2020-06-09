# pytorch-explainable-cnn

## Vanilla Backpropagation

```python
!python vanilla_backprop.py \
    --model alexnet \
    --input data \
    --output output \
    --gpu_id 0
```

![](output/vanilla_backprop_snake.png)

## Guided Backpropagation

```python
!python guided_backprop.py \
    --model alexnet \
    --input data \
    --output output \
    --gpu_id 0
```

![](output/guided_backprop_snake.png)

## GradCAM

```python
!python gradcam.py \
    --model alexnet \
    --input data \
    --output output \
    --target features.11 \
    --gpu_id 0
```

![](output/gradcam_snake.png)

## Guided GradCAM

```python
!python guided_gradcam.py \
    --model alexnet \
    --input data \
    --output output \
    --target features.11 \
    --gpu_id 0
```

![](output/guided_gradcam_snake.png)