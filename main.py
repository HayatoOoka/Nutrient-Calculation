from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import math
import torch
import cv2
import numpy as np
import shutil
import os

app = FastAPI()

# HTMLテンプレートを使うための設定
templates = Jinja2Templates(directory="templates")

def food_sanitation(input_image,weights):
  # モデルの読み込み
  model = torch.hub.load("\\Users\\T21019\\Desktop\\app\\yolov5", "custom", path=weights, source="local")
  # 入力画像の読み込み
  img_np = cv2.imread(input_image)

  # 検出の閾値設定
  model.conf = 0.5

  # 物体検出
  result = model(img_np)

  # バウンディングボックスを取得し画像をクリップ
  for idx, row in enumerate(result.pandas().xyxyn[0].itertuples(), start=1):
      height, width = img_np.shape[:2]

      xmin = math.floor(width * row.xmin)
      xmax = math.floor(width * row.xmax)
      ymin = math.floor(height * row.ymin)
      ymax = math.floor(height * row.ymax)

      area = (xmax-xmin)*(ymax-ymin)

      cripped_img = img_np[ymin:ymax, xmin:xmax]

  return area

def cal_intake_Calculation(first,second,weights):
  burg_cal = 500
  first_area = food_sanitation(first,weights)
  second_area =food_sanitation(second,weights)
  ratio = first_area/second_area
  cal_intake = burg_cal*ratio - burg_cal
  return int(cal_intake)

def get_uploadfile(upload_file: UploadFile): # フロント側のFormDataのkeyに合わせる(upload_file)
    path = f'\\Users\\T21019\\Desktop\\app\\files\\{upload_file.filename}'# api/filesディレクトリを作成しておく
    with open(path, 'wb+') as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return {
        'filename': path,
        'type': upload_file.content_type
    }

@app.get("/", response_class=HTMLResponse)
def home(request: HTMLResponse):
    return templates.TemplateResponse("page.html", {"request": request})

@app.post("/upload")
async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # アップロードされた画像を読み込む
  
  first_image = get_uploadfile(image1)
  second_image = get_uploadfile(image2)
  first_path = first_image['filename']
  second_path = second_image['filename']
  result = cal_intake_Calculation(first_path,second_path,'./best.pt')
  result_html = f"<p>摂取カロリー{result}kcal</p>"
  os.remove(first_path)
  os.remove(second_path)

  return HTMLResponse(content=result_html)