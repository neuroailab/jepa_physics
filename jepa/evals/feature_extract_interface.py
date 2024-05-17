import numpy as np
import torch.nn as nn
from torchvision import transforms


class ActivityRecogFeatureExtractor(nn.Module):

    def __int__(self):
        '''
        weights_path: path to the weights of the model
        '''

    def get_embed_dim(self):
        '''
        :return: Embedding Dimension
        '''


    def get_num_heads(self):
        '''
        :return: Number of Heads
        '''


    def transform(self,):
        '''
        :return: Image Transform
        '''

    def forward(self, videos):
        '''
        videos: [B, C, N, H, W], N is usually 100 for intphys and videos are normalized with imagenet norm
        returns: plausibility score for the video by using reconstruction losses on frame triplets
        '''


class Test(ActivityRecogFeatureExtractor):
    def __init__(self):
        super().__init__()

    def transform(self,):
        '''
        :return: Image Transform
        '''

        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def forward(self, videos):
        '''
        videos: [B, C, N, H, W], N is usually 100 for intphys and videos are normalized with imagenet norm
        returns: plausibility score for the video by using reconstruction losses on frame triplets
        '''

