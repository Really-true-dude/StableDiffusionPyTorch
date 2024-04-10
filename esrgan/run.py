import torch
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Generator, initialize_weights
from model_converter import convert_model_weights

def load_model(checkpoint_path: str, model: torch.nn.Module, device = "cpu"):
    if checkpoint_path == "esrgan/weights/R-ESRGAN_x4.pth":
        checkpoint = convert_model_weights(checkpoint_path, device=device)
    elif checkpoint_path == "esrgan/weights/ESRGAN_gen.pth":
        checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)
    else:
        raise Exception("Unsupported model dict.. Try downloading one of the models provided on git repository..")
    
    model.load_state_dict(checkpoint["params_ema"], strict = True)

def upscale(save_path: str, gen: torch.nn.Module, image_path: str, device):

    transform = A.Compose([
        A.Normalize(mean=(0, 0, 0), std = (1, 1, 1)),
        ToTensorV2()
    ])

    gen.eval()
    image = Image.open(image_path)

    with torch.no_grad():
        upscaled_img = gen(transform(image=np.asarray(image))["image"].unsqueeze(0).to(device))

    save_image(upscaled_img, save_path + "upscaled.jpg")
    

if __name__ == "__main__":

    SAVE_PATH = "esrgan/output/"
    IMAGE_PATH = "esrgan/01.jpg"
    ALLOW_CUDA = False
    DEVICE = "cuda" if torch.cuda.is_available() and ALLOW_CUDA else "cpu"
    MODEL_PATH = "esrgan/weights/R-ESRGAN_x4.pth"


    gen = Generator(in_channels=3).to(DEVICE)

    initialize_weights(gen)
    load_model(MODEL_PATH, gen, DEVICE)

    upscale(SAVE_PATH, gen, IMAGE_PATH, DEVICE)
    
