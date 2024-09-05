# import os


# from utils.config import Config
# config_file = "/root/code/LLM_pipline/video_chat2/configs/config_mistral_hd.json"
# cfg = Config.from_file(config_file)
# import io

# from models import VideoChat2_it_hd_mistral
# from utils.easydict import EasyDict
# import torch

# from transformers import StoppingCriteria, StoppingCriteriaList

# from PIL import Image
# import numpy as np
# import numpy as np
# from decord import VideoReader, cpu
# import torchvision.transforms as T
# from torchvision.transforms import PILToTensor
# from torchvision import transforms
# from dataset.video_transforms import (
#     GroupNormalize, GroupScale, GroupCenterCrop, 
#     Stack, ToTorchFormatTensor
# )
# from torch.utils.data import Dataset
# from torchvision.transforms.functional import InterpolationMode

# from torchvision import transforms

# import matplotlib.pyplot as plt

# from IPython.display import Video, HTML

# from peft import get_peft_model, LoraConfig, TaskType
# import copy

# import json
# from collections import OrderedDict

# from tqdm import tqdm

# import decord
# import time
# decord.bridge.set_bridge("torch")

# cfg.model.vision_encoder.num_frames = 4
# model = VideoChat2_it_hd_mistral(config=cfg.model)
import torch
torch.cuda.empty_cache()