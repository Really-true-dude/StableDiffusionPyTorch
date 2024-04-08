import torch
from torchvision.utils import save_image
from model import Generator, initialize_weights
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(checkpoint_path: str, model: torch.nn.Module):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only = False)
    model.load_state_dict(checkpoint["state_dict"], strict = True)

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

    SAVE_PATH = "output/"
    ALLOW_CUDA = True
    DEVICE = "cuda" if torch.cuda.is_available() and ALLOW_CUDA else "cpu"
    MODEL_PATH = "ESRGAN_gen.pth"

    gen = Generator(in_channels=3).to(DEVICE)

    initialize_weights(gen)
    load_model(MODEL_PATH, gen)

    upscale(SAVE_PATH, gen, "img.jpg", DEVICE)
    
