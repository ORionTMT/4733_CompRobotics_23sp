from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
 
        obj_angle = data['angle'].item()
        angle_bins = np.linspace(0, 180, num=9)
        angle_index = np.digitize(obj_angle, angle_bins) - 1


        center_pt = data['center_point'].numpy()
        kps = KeypointsOnImage([Keypoint(x=center_pt[0], y=center_pt[1])], shape=data['rgb'].shape)


        rotation_angle = angle_index * 22.5
        rotator = iaa.Rotate(rotation_angle)
        rotated_image = rotator.augment_image(data['rgb'].numpy())
        rotated_kps = rotator.augment_keypoints(kps)

        input_tensor = torch.from_numpy(rotated_image).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1)

        rotated_center_pt = np.array([rotated_kps.keypoints[0].x, rotated_kps.keypoints[0].y], dtype=np.float32)
        target_scoremap = get_gaussian_scoremap(input_tensor.shape[1:], rotated_center_pt, sigma=1)
        target_tensor = torch.from_numpy(target_scoremap)[None, ...]

        return {'input': input_tensor, 'target': target_tensor}

class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
            self, 
            rgb_obs: np.ndarray,  
        ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        angles = np.arange(8) * 22.5
        h, w, c = rgb_obs.shape
        rotated_image_batch = torch.zeros((8, c, h, w))  


        for i, angle in enumerate(angles):
            seq = iaa.Sequential([iaa.Affine(rotate=angle)])
            image_aug = seq(image=rgb_obs)
            image = torch.from_numpy(image_aug)
            rotated_image_batch[i] = torch.permute(image, (2, 0, 1))

        rotated_image_batch = rotated_image_batch/255


        data = rotated_image_batch.to(device) 
        output = self.predict(data)
        pred = output.detach().to('cpu').numpy() 
        max_affordance = 0
        grasp_x, grasp_y, grasp_angle = 0, 0, 0

        for i, output in enumerate(pred):
            output = output.squeeze()
            x, y = np.unravel_index(output.argmax(), output.shape)
            if output[x][y] > max_affordance:
                max_affordance = output[x][y]
                grasp_x, grasp_y, grasp_angle = y, x, -angles[i]


        seq = iaa.Sequential([iaa.Affine(rotate=grasp_angle)])
        kps = KeypointsOnImage([Keypoint(x=grasp_x, y=grasp_y)], shape=(h, w))

        kps_aug = seq(keypoints=kps)
        coord, angle = (int(kps_aug.keypoints[0].x), int(kps_aug.keypoints[0].y)), grasp_angle

        for max_coord in list(self.past_actions): 
            bin = max_coord[0] 
            suppression_map = get_gaussian_scoremap(shape=pred[bin].shape, keypoint=np.array(max_coord[1], dtype=np.float32), sigma=4)
            pred[bin] -= suppression_map

        max_coord_idx = np.argmax([output.max() for output in pred])
        max_coord = (max_coord_idx, np.unravel_index(pred[max_coord_idx].argmax(), pred[max_coord_idx].shape))
        self.past_actions.append(max_coord)

        # Assuming bin is the same for all max_coords

        # ===============================================================================
        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        combined_images = []

        for i in range(0, len(rotated_image_batch), 2):
            # visualize rgb image and affordance for batch_index = even
            rotated_input = rotated_image_batch[i].numpy()
            
            if angles[i] == -grasp_angle:
                rotated_input = np.transpose(rotated_input, (1, 2, 0))
                rotated_input = np.ascontiguousarray(rotated_input*255, dtype=np.uint8)
                draw_grasp(rotated_input, (grasp_x, grasp_y), 0)
                rotated_input = np.transpose(rotated_input/255, (2, 0, 1))

            rgb_and_affordance_left = self.visualize(rotated_input, pred[i]) 


            rotated_input = rotated_image_batch[i+1].numpy()
            
            if angles[i+1] == -grasp_angle:
                rotated_input = np.transpose(rotated_input, (1, 2, 0))
                rotated_input = np.ascontiguousarray(rotated_input*255, dtype=np.uint8)
                draw_grasp(rotated_input, (grasp_x, grasp_y), 0)
                rotated_input = np.transpose(rotated_input/255, (2, 0, 1))

            rgb_and_affordance_right = self.visualize(rotated_input, pred[i+1])  
            combined_image = np.concatenate((rgb_and_affordance_left, rgb_and_affordance_right), axis=1)
            combined_image[-1] = 127
            combined_images.append(combined_image)

        vis_img = np.concatenate(combined_images, axis=0) 
        # ===============================================================================
        return coord, angle, vis_img


        

