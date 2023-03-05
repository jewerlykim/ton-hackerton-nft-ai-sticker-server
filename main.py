import time
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

# model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
# version = model.versions.get("e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")
# model = replicate.models.get("cjwbw/stable-diffusion-v2-inpainting")
# version = model.versions.get("f9bb0632bfdceb83196e85521b9b55895f8ff3d1d3b487fd1973210c0eb30bec")
model = replicate.models.get("stability-ai/stable-diffusion-inpainting")
version = model.versions.get("c28b92a7ecd66eee4aefcd8a94eb9e7f6c3805d5f06038165407fb5cb355ba67")


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


@app.post("/batch_predict")
async def batch_predict(
    image: UploadFile = File(...),
    project_name: str = Form(...)
    ):
    print("batch_predict called with project_name: ", project_name)

    # 7가지 상황에 맞는 prompt 배열
    # 기쁨: 노란색 배경과 밝은 눈, 넓게 웃는 입으로 캐릭터를 표현할 수 있습니다.
    # 슬픔: 캐릭터를 푹신한 검은색 배경과 울고 있는 눈, 슬픈 표정으로 표현할 수 있습니다.
    # 분노: 빨간색 배경과 찡그린 눈, 큰 입으로 분노한 표정을 표현할 수 있습니다.
    # 놀람: 푹신한 하늘색 배경과 큰 눈, 입을 벌리고 놀라는 표정으로 캐릭터를 표현할 수 있습니다.
    # 사랑: 분홍색 배경과 큰 눈, 미소짓는 입으로 캐릭터를 표현할 수 있습니다.
    # 축하: 파란색 배경과 큰 눈, 손에 선물 상자를 들고 있는 표정으로 캐릭터를 표현할 수 있습니다.
    # 신나는: 오렌지색 배경과 광대한 눈, 미소 짓고 손을 흔드는 표정으로 캐릭터를 표현할 수 있습니다.
    base_positive_prompts = "(pixar style:1.0), 8k, High Detail, 3D, (one girl:2.0),(one person:2.0), simple hair, no hair band, (girl:1.0)"
    base_negative_prompts = "disfigured, bad art, extra fingers, mutated hands, blurry, bad anatomy, bad hair, arms, Accessories, (hair band:1.0), hat, hoodie, cap, glowing hair, people"
    prompts = [
        # ("(single color background:2.0),(Wide smile:2.0), (smile with tooth:1.0), (feminine clothes:2.0), (long straight hair:2.0),(untied hair:2.0),(black hair:2.0), (feminine hair:2.0)", "(bad quility:2.0)"),
        ("(single color background:1.5),(pursed Lips:2.0),(sad lips:2.0),(the corners of one's mouth dropping:2.0), (feminine clothes:2.0),(cloth:2.0), (long straight hair:1.0),(untied hair:1.0),(black hair:1.0), (feminine hair:1.0)", "(bad quility:1.0)"),
        
    ]

    results = []

    for prompt, negative_prompt in prompts:
        inputs = {
            'prompt': prompt + ", " + base_positive_prompts,
            'negative_prompt': negative_prompt + ", " + base_negative_prompts,
            'image': io.BytesIO(image.file.read()),
            'mask': open("assets/clonex_mask.png", "rb"),
            'invert_mask': False,
            'num_outputs': 1,
            'num_inference_steps': 70,
            'guidance_scale': 9,
        }
        # 100, 9

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
            results.extend(outputs)
            # results.append(outputs)
        
    return results


@app.post("/batch_predict_test")
async def batch_predict_test(
    image: UploadFile = File(...),
    project_name: str = Form(...)
    ):
    print("batch_predict called with project_name: ", project_name)

    # 7가지 상황에 맞는 prompt 배열
    # 기쁨: 노란색 배경과 밝은 눈, 넓게 웃는 입으로 캐릭터를 표현할 수 있습니다.
    # 슬픔: 캐릭터를 푹신한 검은색 배경과 울고 있는 눈, 슬픈 표정으로 표현할 수 있습니다.
    # 분노: 빨간색 배경과 찡그린 눈, 큰 입으로 분노한 표정을 표현할 수 있습니다.
    # 놀람: 푹신한 하늘색 배경과 큰 눈, 입을 벌리고 놀라는 표정으로 캐릭터를 표현할 수 있습니다.
    # 사랑: 분홍색 배경과 큰 눈, 미소짓는 입으로 캐릭터를 표현할 수 있습니다.
    # 축하: 파란색 배경과 큰 눈, 손에 선물 상자를 들고 있는 표정으로 캐릭터를 표현할 수 있습니다.
    # 신나는: 오렌지색 배경과 광대한 눈, 미소 짓고 손을 흔드는 표정으로 캐릭터를 표현할 수 있습니다.
    base_positive_prompts = "(anime),(pixar style:2.0), 8k, High Detail, 3D, (one girl:2.0),(one person:2.0), simple hair, no hair band"
    base_negative_prompts = "disfigured, bad art, extra fingers, mutated hands, blurry, bad anatomy, bad hair, arms, Accessories, (hair band:1.0), hat, hoodie, cap, glowing hair"
    prompts = [
        ("(sky:2.0),(Wide smile:2.0), (smile with tooth:1.0), (white shirt:1.0), (long straight hair:2.0),(untied hair:2.0),(black hair:2.0)", "(bad quility:2.0)"),
        # ("black color background, pouty mouth, white color shirt, gold color hair", "bad quility"),
        
    ]

    results = []

    for prompt, negative_prompt in prompts:
        inputs = {
            'prompt': prompt + ", " + base_positive_prompts,
            'negative_prompt': negative_prompt + ", " + base_negative_prompts,
            'image': io.BytesIO(image.file.read()),
            'mask': open("assets/clonex_mask.png", "rb"),
            'invert_mask': False,
            'num_outputs': 1,
            'num_inference_steps': 100,
            'guidance_scale': 7.5,
            'prompt_strength': 0.8,

        }

        outputs = version.predict(**inputs)
        print(outputs)
        if outputs:
            # os.makedirs("outputs", exist_ok=True)

            # # Get the latest file number in the output directory
            # existing_files = os.listdir('outputs')
            # latest_file_num = max([0] + [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith(project_name)])
            # next_file_num = latest_file_num + 1

            # # Save the output image to a file in the output directory
            # output_path = f'outputs/{project_name}_{next_file_num}.png'
            # response = requests.get(outputs[0])
            # with open(output_path, 'wb') as f:
            #     f.write(response.content)

            # Return the output image as a response
            results.append(outputs[0])
            results.append(outputs[0])
            results.append(outputs[0])
        
    return results

# x 초를 key로 받아서 key 초 만큼 기다렸다가 응답해줌
@app.post("/timelock")
async def timelock(x: int):
    print("timelock called with x: ", x)
    time.sleep(x)
    return {"result": "success"}
