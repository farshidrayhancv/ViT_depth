# ViT_depth

This repository contains code for testing the Vision Transformer (ViT) model with depth/4th channel for three different tasks:

1. Image Classification
2. Single Object Detection
3. Multi Object Detection

## Sample Data

The repository provides sample data for each task. Here are the sample images:

1. Sample Data for Single RGBD Image from Random Noise with Depth:
    ![multi_class_objdet_data](https://github.com/farshidrayhanuiu/ViT_depth/blob/main/etc/RGBD_random_noise_image_sample.png)

2. Sample Data for Single Class Object Detection:
    ![multi_class_objdet_data](https://github.com/farshidrayhanuiu/ViT_depth/blob/main/etc/single_class_objdet_data.png.png)

3. Sample Data for Multi Class Object Detection (up to 5 classes):
    ![multi_class_objdet_data](https://github.com/farshidrayhanuiu/ViT_depth/blob/main/etc/multi_class_objdet_data.png)

## Model Architecture

### RGBDViT

The RGBDViT model takes RGB and depth images as input. The input images are preprocessed by flattening and concatenating the RGB and depth channels into a single vector. This vector represents the input sequence of tokens.

The concatenated vector is then passed through an embedding layer to project it into the desired embedding size, capturing important features from the input sequence.

The embedded sequence is reshaped to have dimensions (batch_size, 1, sequence_length) and passed through a series of transformer blocks. These blocks allow each position to attend to other positions, processing the sequence.

After the transformer blocks, global average pooling is applied to the output sequence, aggregating information from different positions and reducing the sequence length to 1.

Finally, the pooled output is passed through fully connected layers to produce predictions for image classification.

### ObjectDetectionViT

The ObjectDetectionViT model is designed for single object detection. It takes an RGB image as input. The input image is preprocessed by flattening it into a single vector.

The vector is then passed through an embedding layer to project it into the desired embedding size, capturing important features from the input image.

The embedded vector is reshaped to have dimensions (batch_size, 1, sequence_length) and passed through a series of transformer blocks. These blocks allow each position to attend to other positions, processing the sequence.

After the transformer blocks, global average pooling is applied to the output sequence, aggregating information from different positions and reducing the sequence length to 1.

Finally, the pooled output is passed through fully connected layers to produce predictions for bounding boxes and class scores.

### MultiObjectDetectionViT

The MultiObjectDetectionViT model is designed for multi-object detection. It takes an RGB image as input. The input image is preprocessed by flattening it into a single vector.

The vector is then passed through an embedding layer to project it into the desired embedding size, capturing important features from the input image.

The embedded vector is reshaped to have dimensions (batch_size, 1, sequence_length) and passed through a series of transformer blocks. These blocks allow each position to attend to other positions, processing the sequence.

After the transformer blocks, global average pooling is applied to the output sequence, aggregating information from different positions and reducing the sequence length to 1.

Finally, the pooled output is passed through fully connected layers to produce predictions for bounding boxes and class scores.

Please refer to the code in this repository for more details on the implementation.
