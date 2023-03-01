from fastapi import FastAPI
import replicate

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
async def predict():
    inputs = {
        # Input prompt
    'prompt': "a herd of grazing sheep",

    # The prompt or prompts not to guide the image generation. Ignored
    # when not using guidance (i.e., ignored if `guidance_scale` is less
    # than `1`).
    'negative_prompt': "",

    # Input image to in-paint. Width and height should both be divisible
    # by 8. If they're not, the image will be center cropped to the
    # nearest width and height divisible by 8
    'image': open("assets/sheep.png", "rb"),

    # Black and white image to use as mask. White pixels are inpainted and
    # black pixels are preserved.
    'mask': open("assets/mask.png", "rb"),

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
    return outputs