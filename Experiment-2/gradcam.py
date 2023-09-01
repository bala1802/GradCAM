import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image


def overlay_gradcam_on_image(model, target_layer, images, preprocess_args, **kwargs):
    rows, cols = int(len(images) / 5), 5
    figure = plt.figure(figsize=(cols*2, rows*2))

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available())
    cam.batch_size = 32

    for i, img in enumerate(images):
        rgb_img = np.float32(img) / 255
        input_tensor = preprocess_image(rgb_img, **preprocess_args)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None, **kwargs)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(cam_image, cmap="rainbow")
    
    plt.tight_layout()
    plt.show()
        