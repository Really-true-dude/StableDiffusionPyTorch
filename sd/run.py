import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
from safetensors.numpy import load_file

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
# elif(torch.has_mps or torch.backend.mps.is_available()) and ALLOW_MPS:
#     DEVICE = "mps"

print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer(vocab_file="data/vocab.json" , merges_file="data/merges.txt")
model_file = "data\kawaiiRealisticAnime_a05.safetensors"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

WIDTH = 512
HEIGHT = 768

## TEXT TO IMAGE
prompt = "1girl, blonde hair, short hair, bun, blue eyes, red blazer, white shirt, blue skirt, 8k, masterpiece"
uncond_prompt = "bad resolution, worst quality, extra digits, ugly"
do_cfg = True
cfg_scale = 7

## IMAGE TO IMAGE
input_image = None
image_path = ""
# input_image = Image.open(image_path)
strength = 0.8

sampler = "ddpm"
num_inference_steps = 30
seed = None

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    HEIGHT=HEIGHT,
    WIDTH=WIDTH,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer
)

image = Image.fromarray(output_image).save("img.jpg")