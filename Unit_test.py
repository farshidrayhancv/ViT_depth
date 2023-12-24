
import unittest
import torch
from src.RGBD_ObjDet_mcls_helper import MultiObjectDetectionViT
from src.RGBD_ObjDet_helper import ObjectDetectionViT


class TestMultiObjectDetectionViT(unittest.TestCase):
    """
    Unit tests for the MultiObjectDetectionViT class.
    """

    def setUp(self):
        """
        Set up the test environment by initializing the model and device.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiObjectDetectionViT(
            channels=3, image_size=224, embed_size=256, num_heads=8, num_classes=10, max_objects=5, depth_channels=1)
        self.model.to(self.device)

    def test_forward_pass(self):
        """
        Test the forward pass of the MultiObjectDetectionViT model.
        """
        batch_size = 8
        dummy_rgb = torch.randn((batch_size, 3, 224, 224))
        dummy_depth = torch.randn((batch_size, 1, 224, 224))

        dummy_rgb = dummy_rgb.to(self.device)
        dummy_depth = dummy_depth.to(self.device)

        with torch.no_grad():
            bbox_preds, class_preds = self.model(dummy_rgb, dummy_depth)

        expected_bbox_shape = (batch_size, 5, 4)
        expected_class_shape = (batch_size, 5, 10)
        self.assertEqual(bbox_preds.shape, expected_bbox_shape)
        self.assertEqual(class_preds.shape, expected_class_shape)


class TestObjectDetectionViT(unittest.TestCase):
    """
    Unit tests for the ObjectDetectionViT class.
    """

    def setUp(self):
        """
        Set up the test environment by initializing the model and device.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = ObjectDetectionViT(
            channels=3, image_size=224, embed_size=256, num_heads=8, num_classes=10, depth_channels=1)
        self.model.to(self.device)

    def test_forward_pass(self):
        """
        Test the forward pass of the ObjectDetectionViT model.
        """
        batch_size = 8
        dummy_rgb = torch.randn((batch_size, 3, 224, 224))
        dummy_depth = torch.randn((batch_size, 1, 224, 224))

        dummy_rgb = dummy_rgb.to(self.device)
        dummy_depth = dummy_depth.to(self.device)

        with torch.no_grad():
            bbox_preds, class_preds = self.model(dummy_rgb, dummy_depth)

        expected_bbox_shape = (batch_size, 4)
        expected_class_shape = (batch_size, 10)
        self.assertEqual(bbox_preds.shape, expected_bbox_shape)
        self.assertEqual(class_preds.shape, expected_class_shape)


if __name__ == '__main__':
    unittest.main()
