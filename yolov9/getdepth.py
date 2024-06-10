from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from layers import disp_to_depth
import networks
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str, help='path to a test image or folder of images', required=True)
    parser.add_argument('--load_weights_folder', type=str, help='path of a pretrained model to use', required=True)
    parser.add_argument('--model', type=str, help='name of a pretrained model to use', default="lite-mono",
                        choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
    parser.add_argument('--ext', type=str, help='image extension to search for in folder', default="jpg")
    parser.add_argument('--no_cuda', action='store_true', help='if set, disables CUDA')
    parser.add_argument('--x', type=float, help='x coordinate', required=True)
    parser.add_argument('--y', type=float, help='y coordinate', required=True)

    return parser.parse_args()

def get_depth_at_pixel(depth_map, x, y):
    return depth_map[y, x]

def load_model(args, device):
    """Load the model based on the provided arguments"""
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    encoder = networks.LiteMono(model=args.model, height=feed_height, width=feed_width)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_width, feed_height

def get_depth_value(image_path, load_weights_folder, model_name, x, y, no_cuda=False):
    class Args:
        def __init__(self):
            self.image_path = image_path
            self.load_weights_folder = load_weights_folder
            self.model = model_name
            self.ext = 'jpg'
            self.no_cuda = no_cuda
            self.x = x
            self.y = y

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    encoder, depth_decoder, feed_width, feed_height = load_model(args, device)

    # Load image and preprocess
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
    depth_map = depth.cpu().numpy().squeeze()

    # Convert float coordinates to integer pixel values
    x_pixel = int(x * depth_map.shape[1])
    y_pixel = int(y * depth_map.shape[0])

    if x_pixel < 0 or x_pixel >= depth_map.shape[1] or y_pixel < 0 or y_pixel >= depth_map.shape[0]:
        raise IndexError(f"Calculated pixel coordinates ({x_pixel}, {y_pixel}) are out of bounds for depth map of shape {depth_map.shape}")

    depth_value = get_depth_at_pixel(depth_map, x_pixel, y_pixel)

    return depth_value


if __name__ == '__main__':
    args = parse_args()
    depth_value = get_depth_value(args.image_path, args.load_weights_folder, args.model, args.x, args.y, args.no_cuda)
    print(f"Depth at ({args.x}, {args.y}) is {depth_value:.2f} meters")
