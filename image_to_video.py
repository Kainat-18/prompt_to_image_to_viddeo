import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from torchvision.io import write_video
import torchvision.transforms as T
import os

# Config
IMAGE_PATH = "viking_jb_image.jpg"
OUTPUT_PATH = "output/viking_video.mp4"
FRAME_SIZE = (576, 320)  # Required size for model
FPS = 7

# 1. Check device (force CPU, because float16 on CPU causes crash)
device = "cpu"

# 2. Load image
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")
init_image = Image.open(IMAGE_PATH).convert("RGB").resize(FRAME_SIZE)

# 3. Load model without float16
print("⏳ Loading model on CPU (no float16)...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float32  # Use float32 for CPU
).to(device)

# 4. Generate video frames
print("🎥 Generating video frames...")
result = pipe(init_image, decode_chunk_size=8, generator=torch.manual_seed(42))
video_frames = result.frames[0]

# 5. Convert frames to video tensor
video_tensor = torch.stack([T.ToTensor()(frame) * 255 for frame in video_frames])
video_tensor = video_tensor.permute(0, 2, 3, 1).to(torch.uint8)

# 6. Write video
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
write_video(OUTPUT_PATH, video_tensor, fps=FPS)
print(f"✅ Video saved to: {OUTPUT_PATH}")
