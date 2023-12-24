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

The input images are preprocessed by flattening and concatenating the RGB and depth channels into a single vector. This vector represents the input sequence of tokens.

The concatenated vector is then passed through an embedding layer to project it into the desired embedding size, capturing important features from the input sequence.

The embedded sequence is reshaped to have dimensions (batch_size, 1, sequence_length) and passed through a series of transformer blocks. These blocks allow each position to attend to other positions, processing the sequence.

After the transformer blocks, global average pooling is applied to the output sequence, aggregating information from different positions and reducing the sequence length to 1.

Finally, the pooled output is passed through fully connected layers to produce predictions for bounding boxes and class scores.

Please refer to the code in this repository for more details on the implementation.

The pooled output is then passed through fully connected layers (self.fc_bbox and self.fc_class) to produce predictions for bounding boxes and class scores.

For each stage of the RGBDViT, ObjectDetectionViT, and MultiObjectDetectionViT models, the vector shape and size can be described as follows:

1. RGBDViT:
   - Input Vector Shape: (batch_size, height, width, 4)
   - Input Vector Size: height x width x 4
   - Embedded Sequence Shape: (batch_size, sequence_length, embedding_size)
   - Embedded Sequence Size: sequence_length x embedding_size
   - Output Sequence Shape: (batch_size, 1, sequence_length)
   - Output Sequence Size: sequence_length

2. ObjectDetectionViT:
   - Input Vector Shape: (batch_size, height, width, 3)
   - Input Vector Size: height x width x 3
   - Embedded Sequence Shape: (batch_size, sequence_length, embedding_size)
   - Embedded Sequence Size: sequence_length x embedding_size
   - Output Sequence Shape: (batch_size, 1, sequence_length)
   - Output Sequence Size: sequence_length

3. MultiObjectDetectionViT:
   - Input Vector Shape: (batch_size, height, width, 3)
   - Input Vector Size: height x width x 3
   - Embedded Sequence Shape: (batch_size, sequence_length, embedding_size)
   - Embedded Sequence Size: sequence_length x embedding_size
   - Output Sequence Shape: (batch_size, 1, sequence_length)
   - Output Sequence Size: sequence_length

Please note that the we use values from original Vit for `batch_size`, `height`, `width`, `sequence_length`, and `embedding_size` will depend on the specific implementation and configuration of the models.


