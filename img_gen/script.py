# off_controlnet_inpaint.py
import torch, json, cv2, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
)

ROOT = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # ---------- û prompt / þ / mask ----------
    wf = json.load(open(ROOT / "workflow.json"))
    prompt = wf["6"]["inputs"]["text"]
    negative = wf["7"]["inputs"]["text"]

    init_img = Image.open(ROOT / "spcar.png").convert("RGB").resize((512, 512))

    # ( Canny Z¹þ JSON Âp	
    canny = cv2.Canny(np.array(init_img), 100, 200)
    control_image = Image.fromarray(canny)

    # Uéb mask:128 Ï ¹F	
    mask = Image.new("L", (512, 512), 255)
    mask.paste(0, (128, 128, 384, 384))  # -Ãéb inpaint

    # ----------  }! ----------
    controlnet = ControlNetModel.from_pretrained(
        ROOT / "offline_models/control_v11p_sd15_canny",  # <-- t*îU
        local_files_only=True, 
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        ROOT / "../models/sd_ck/",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True, 
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ---------- ¨ ----------
    with torch.autocast(device):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init_img,
            #mask_image=mask,
            control_image=control_image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator(device).manual_seed(556676558500698),
        ).images[0]

    out.save(ROOT / "output.png")
    print("done  output.png")

if __name__ == "__main__":
    main()