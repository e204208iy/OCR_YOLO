import pandas as pd
import os
import shutil
import glob
import cv2
from ultralytics import YOLO
import easyocr
import PIL.ExifTags as ExifTags
from PIL import Image
import datetime

numDF = pd.DataFrame(columns=['ImageName', 'num'])
img_folder_path = "./TargetFolder/"
img_file_list = glob.glob(os.path.join(img_folder_path,"*.jpg"))

model = YOLO(model="./runs/detect/train2/weights/last.pt")
reader = easyocr.Reader(['en'])

if not os.path.isdir("output"):
    os.makedirs("output")
if not os.path.isdir("renamedImages"):
    os.makedirs("renamedImages")

#CSVとそのファイル名を作成するためのリスト
detectNumList = []
detectStringList = []
datetimeList =[]

for img_path in img_file_list:
    img = cv2.imread(img_path)
    img_PIL = Image.open(img_path)
    OCR_results = reader.readtext(img)
    ObjectDetection_results = model.predict(img,save=True)
    
    detectString = ""
    #OCRで検出された文字抽出、リストに格納
    for result in OCR_results:
        n=result[1]
        detectString+=n
    detectStringList.append(detectString)
    print("detectString",detectString)
    
    #きゅうりの本数をリストに格納
    for result in ObjectDetection_results:
        detectNumList.append(len(result))
        
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_PIL._getexif().items() if k in ExifTags.TAGS}
    if "DateTimeOriginal" in exif_dict:
        #撮影日時に基づく新規ファイル名を準備
        file_dateTime = datetime.datetime.strptime(exif_dict["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
        file_dateTime = file_dateTime.strftime("%Y-%m-%d_%H-%M-%S")
        datetimeList.append(file_dateTime)
    else:
        #DateTimeOriginalが存在しなかった場合
        datetimeList.append("2023-0-0")
    img_PIL.close()

print("OCRで検出された文字列の数",len(detectStringList))
print("元画像のメタデータ（datetime）の数",len(datetimeList))

#画像ファイルのファイル名変更
ChangeFileNameList =[]
for index in range(len(detectStringList)):
    character = detectStringList[index] + "_" + datetimeList[index] + ".jpg"
    ChangeFileNameList.append(character)
  
#元画像を残したまま新しいディレクトリに名前変更済みの画像を格納
# contents = os.listdir(img_folder_path)
# for item in contents:
#     source_item = os.path.join(img_folder_path, item)
#     destination_item = os.path.join("./renamedImages", item)
#     shutil.copy2(source_item, destination_item)
# for index,img in enumerate(img_file_list):
#     os.rename(img,"./renamedImages/" + detectStringList[index] + "_" + datetimeList[index] + ".jpg")
    

#結果をCSVに出力
result_list = [[item1, item2] for item1, item2 in zip(ChangeFileNameList, detectNumList)]
numDF = pd.DataFrame(result_list, columns=numDF.columns)
numDF.to_csv('output/cucumber_num.csv', index=False)