import uuid
import fastapi
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
import cv2
import config
from PIL import Image
import inference
from typing import Dict
from concurrent.futures import ProcessPoolExecutor
import asyncio
from asyncio import AbstractEventLoop
from functools import partial
import time


async def generate_remaining_models(models, image, name: str):
    executor: ProcessPoolExecutor = ProcessPoolExecutor()
    event_loop: AbstractEventLoop = asyncio.get_event_loop()
    await event_loop.run_in_executor(executor,
                                     partial(process_image, models, image, name))


def process_image(models: Dict, image: np.array, name: str):
    for model in models:
        try:
            output, resized = inference.inference(model, image)
            name = name.split(".")[0]
            name = f"{name.split('_')[0]}_{models[model]}.jpg"
            cv2.imwrite(name, output)
        except Exception as e:
            print(e)


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "welcome to the api"}


@app.post("/{style}")
async def get_image(style: str, file: UploadFile = File(...)):
    image: np.array = np.array(Image.open(file.file))
    model: str = config.STYLES[style]
    start: float = time.time()
    output, resized = inference.inference(model, image)
    name: str = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    models: Dict = config.STYLES.copy()
    del models[style]
    asyncio.create_task(generate_remaining_models(models, image, name))
    return {"name": name, "time": time.time() - start}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
