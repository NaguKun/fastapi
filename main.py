from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import os
import random
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

app = FastAPI()

STABLE_DIFFUSION_API = "http://127.0.0.1:7860"

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "(worst quality, low quality, jpeg artifacts)"
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None
    return_base64: Optional[bool] = False

@app.post("/generate")
def generate_image(data: GenerationRequest):
    seed = data.seed if data.seed is not None else random.randint(1, 1_000_000)

    payload = {
        "prompt": data.prompt,
        "negative_prompt": data.negative_prompt,
        "width": data.width,
        "height": data.height,
        "seed": seed,
    }

    try:
        response = requests.post(f"{STABLE_DIFFUSION_API}/sdapi/v1/txt2img", json=payload)
        response.raise_for_status()
        r = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting diffusion server: {e}")

    if 'images' not in r:
        raise HTTPException(status_code=500, detail=f"No images returned from API. Response: {r}")

    # Save image
    os.makedirs("generated_images", exist_ok=True)
    filename = f"generated_images/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(r['images'][0]))

    result = {
        "message": "✅ Image generated successfully",
        "file": filename,
        "seed": seed,
    }

    if data.return_base64:
        result["base64_image"] = r['images'][0]

    return result

def raw_b64_img(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# === ControlNetUnit Class ===
class ControlNetUnit:
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        mask: Optional[Image.Image] = None,
        module: str = "none",
        model: str = "None",
        weight: float = 1.0,
        resize_mode: str = "Resize and Fill",
        low_vram: bool = False,
        processor_res: int = 512,
        threshold_a: float = 64,
        threshold_b: float = 64,
        guidance_start: float = 0.0,
        guidance_end: float = 1.0,
        control_mode: int = 0,
        pixel_perfect: bool = False,
        guessmode: Optional[int] = None,
        hr_option: str = "Both",
        enabled: bool = True,
    ):
        self.image = image
        self.mask = mask
        self.module = module
        self.model = model
        self.weight = weight
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.enabled = enabled

        if guessmode is not None:
            print("⚠️  guessmode is deprecated. Use control_mode instead.")
            control_mode = guessmode

        # Convert control_mode integer to the correct string value
        if control_mode == 0:
            self.control_mode = 'Balanced'
        elif control_mode == 1:
            self.control_mode = 'My prompt is more important'
        elif control_mode == 2:
            self.control_mode = 'ControlNet is more important'
        else:
            self.control_mode = 'Balanced'  # Default to Balanced if invalid value

        self.pixel_perfect = pixel_perfect
        self.hr_option = hr_option

    def to_dict(self):
        return {
            "input_image": raw_b64_img(self.image) if self.image else "",
            "mask": raw_b64_img(self.mask) if self.mask else None,
            "module": self.module,
            "model": self.model,
            "weight": self.weight,
            "resize_mode": self.resize_mode,
            "low_vram": self.low_vram,
            "processor_res": self.processor_res,
            "threshold_a": self.threshold_a,
            "threshold_b": self.threshold_b,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "control_mode": self.control_mode,
            "pixel_perfect": self.pixel_perfect,
            "hr_option": self.hr_option,
            "enabled": self.enabled,
        }
@app.post("/generate-controlnet")
# === Generate image with ControlNet and return result as dict ===
def generate_image_with_controlnet_v2(
    image_path,
    prompt,
    negative_prompt="(low quality, bad anatomy)",
    controlnet_model="control_v11p_sd15_lineart [43d4be0d]",
    module="lineart_realistic",
    output_path="output/controlnet_result.png",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    seed: int = -1,
    cfg_scale: float = 7.0,
    sampler_index: str = "Euler a"
):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Set up ControlNet unit
    control_unit = ControlNetUnit(
        image=image,
        module=module,
        model=controlnet_model,
        weight=1.0,
        resize_mode="Resize and Fill",
        control_mode=0,
        pixel_perfect=True,
        guidance_start=0.0,
        guidance_end=1.0,
    )

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "sampler_index": sampler_index,
        "cfg_scale": cfg_scale,
        "alwayson_scripts": {
            "controlnet": {
                "args": [control_unit.to_dict()]
            }
        }
    }

    # Call Stable Diffusion WebUI API
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
    response.raise_for_status()
    result = response.json()

    # Decode and save image
    img_data = base64.b64decode(result["images"][0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_data)

    return {
        "message": "✅ ControlNet Image Generated",
        "image_path": output_path,
        "base64_image": result["images"][0]
    }