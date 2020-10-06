import uuid
import fastapi
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
import cv2
from PIL import Image

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "welcome to the api"}

@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = None
    output, resized = None, None

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

