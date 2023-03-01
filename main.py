from fastapi import FastAPI, File, UploadFile, Form
import replicate
from pydantic import BaseModel
import io
import os
import requests

class PredictBody(BaseModel):
    prompt: str
    negative_prompt: str
    project_name: str

app = FastAPI()

model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
version = model.versions.get("e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")



@app.get("/")
async def root():
    return {"message": "Hello World"}

# ping
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# predict
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    project_name: str = Form(...)
    ):

    # file size
    print("image size: ", image.file)

    print("predict called with prompt: ", prompt)
    print("predict called with negative_prompt: ", negative_prompt)
    print("predict called with project_name: ", project_name)

    # Load the mask file based on the project name
    # mask_file = f"assets/{project_name}_mask.png"
    # with open(mask_file, "rb") as f:
    #     mask = f.read()

    # if no prompt error
    if prompt == "":
        return {"error": "No prompt provided"}

    inputs = {
        # Input prompt
    'prompt': prompt,

    # The prompt or prompts not to guide the image generation. Ignored
    # when not using guidance (i.e., ignored if `guidance_scale` is less
    # than `1`).
    'negative_prompt': negative_prompt,

    # Input image to in-paint. Width and height should both be divisible
    # by 8. If they're not, the image will be center cropped to the
    # nearest width and height divisible by 8
    'image': io.BytesIO(image.file.read()),

    # Black and white image to use as mask. White pixels are inpainted and
    # black pixels are preserved.
    'mask': open("assets/clonex_mask.png", "rb"),

    # If this is true, then black pixels are inpainted and white pixels
    # are preserved.
    'invert_mask': False,

    # Number of images to output. NSFW filter in enabled, so you may get
    # fewer outputs than requested if flagged
    # Range: 1 to 4
    'num_outputs': 1,

    # Number of denoising steps
    # Range: 1 to 500
    'num_inference_steps': 50,

    # Scale for classifier-free guidance
    # Range: 1 to 20
    'guidance_scale': 7.5,

    # Random seed. Leave blank to randomize the seed
    # 'seed': ...,
    }



    outputs = version.predict(**inputs)
    print(outputs)
    if outputs:
        os.makedirs("outputs", exist_ok=True)

        # Get the latest file number in the output directory
        existing_files = os.listdir('outputs')
        latest_file_num = max([0] + [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith(project_name)])
        next_file_num = latest_file_num + 1

        # Save the output image to a file in the output directory
        output_path = f'outputs/{project_name}_{next_file_num}.png'
        response = requests.get(outputs[0])
        with open(output_path, 'wb') as f:
            f.write(response.content)

        # Return the output image as a response
        return outputs[0]
    return outputs

# simple post test : body로 값이 잘 들어오는지 확인
@app.post("/test")
async def test(
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    project_name: str = Form(...)
    ):
    

    print("test called with prompt: ", prompt)
    print("test called with negative_prompt: ", negative_prompt)
    print("test called with project_name: ", project_name)

    return {"message": "test"}