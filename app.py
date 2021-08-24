# -*- coding: UTF-8 -*-
from flask import Flask, config, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from flask import send_from_directory
from innotis_client import Client
from tis_tools.log import print_title

import cv2
import os
import json
import sys
import numpy as np

import base64 
from PIL import Image
import io
import threading
import time

import random
import string

############################################################################################

# 建立一個 App
app = Flask(__name__)

# 上傳到的資料夾 ( Server端 ) 
UPLOAD_FOLDER = './static'

if not os.path.exists(UPLOAD_FOLDER):
    print_title(f'Create Directory :　[{UPLOAD_FOLDER}]')
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 設定 副檔名 規範
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])

# 設定上傳最大檔案的限制
# max_nums = 16
# app.config['MAX_CONTENT_LENGTH'] = max_nums * 1024 * 1024  # 16MB

client = []
model = None
mode = 'image'
isYolo = 0
seed=1
isStream = False
cap = []
out_pth = ''
info = ''

# 模型資訊
model_list = {  
    'default':{ 'name':'default', 'width':'e.g. 224, 416', 'height':'e.g. 224, 416', 'dataset':'The trainning dataset ( e.g. ImageNet, COCO, Custom )'},
    'densenet_onnx':{ 'name':'densenet_onnx', 'width':224, 'height':224, 'dataset':'imagenet'},
    'yolov4':{ 'name':'yolov4', 'width':608, 'height':608, 'dataset':'coco'},        
    'yolov4_will':{ 'name':'yolov4_will', 'width':608, 'height':608, 'dataset':'will'}
}
# 其他的 client 設定
client_setup = {
    'conf':0.9,
    'nms':0.1,
    'info':False,
    'timeout':False
}

############################################################################################

''' 檢查副檔名：有副檔名 以及 副檔名有在制定的規範內 回傳 mode '''
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

''' 取得模型資訊之字串 '''
def get_info(model):

    info = ''
    [ info + f'{item}: {val}\n' for item, val in model.items() ] 
    return info

'''隨機產生UUID'''
def gen_uuid(img_path):

    global seed
    seed += 1  # 改變種子以取得不同隨機數值
    random.seed(seed)
    
    name, ext = img_path.rsplit('.', 1)
    fakeid='?uuid='
       
    for i in range(3):
        fakeid += '{}{}'.format(random.randint(0,9), string.ascii_letters[random.randint(0,26)])  # 取得 隨機 數值 與 字母

    return '{}{}.{}'.format(name, fakeid, ext)

'''清除資料夾中的圖片'''
def clear_images(dir): 
    [ os.remove(os.path.join(dir, file)) for file in os.listdir(app.config['UPLOAD_FOLDER']) if file.endswith(tuple(ALLOWED_EXTENSIONS)) ]

'''解析辨識結果'''
def parse_results(res):
    
    if isYolo:
        index=0
        det_nums = 0 if res[0][0] is 'None' else len(res)   # 如果第一筆資料是 None 代表沒有辨識到結果

        results ='Detected Objects: {} \n'.format(det_nums)
        for classes, confidence in res.values():
            index+=1
            results = results + '[{}] {} , {:.3f}\n'.format(index, classes, confidence)
    else:
        index, classes, confidence = res[0]
        results = '[{}] {} , {:.3f}\n'.format(index, classes, confidence)

    return results

'''Convet cv to base64'''
def cv2base64(img):
    # cv to pillow
    img = Image.fromarray(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
    
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)        # clear temp
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    return uri

''' Stream Mode '''
def stream_infer():

    global isStream,info
    
    while(isStream):

        ret, frame = cap.read()

        if not ret:
            print_title('out stream') 
            break

        res, image_draw =  client.image_infer(frame, model['width'], model['height'])

        info = res

        ret, jpg = cv2.imencode('.jpg', image_draw )

        yield ( b'--frame\r\n' + 
                b'Content-Type: image/jpeg\r\n\r\n' + 
                jpg.tobytes() + 
                b'\r\n')    

    isStream = False

############################################################################################

'''根目錄 : 通常會放置首頁'''
@app.route('/', methods=['GET', 'POST'])
def index():
    # global models_info, UPLOAD_FOLDER
    global isStream
    isStream = False

    print_title('Clear Temp Dir ... ')
    clear_images(UPLOAD_FOLDER)
    
    # 使用 render_template 會自動去 templates 抓取 index.html
    return render_template('index.html', models=model_list.items())

'''不斷餵新的畫面'''
@app.route("/_feedFrame") 
def feed_frame():
    return Response(stream_infer(), mimetype='multipart/x-mixed-replace; boundary=frame')

'''取得選擇的模型'''
@app.route('/_sel', methods=['GET', 'POST'])
def get_sel():
    global model, model_list, isYolo
    model = model_list[request.get_json()]
    isYolo = 'yolo' in model['name']
    print_title('Selected Model ... {}'.format(model['name']))
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

'''取得選擇到的模式'''
@app.route('/_mode', methods=['GET', 'POST'])
def get_mode():
    global mode
    mode = request.get_json()
    print_title('Selected Mode ... {}'.format(mode))
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

'''註冊 upload 位置'''
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    
    global client, model, img, out_pth, info

    # 如果有 POST 就會產生資料列表
    # if request.method == 'POST':

    # 取得 IP 後建立 Client 端物件
    url = request.form.get('ip')
    client = Client(    url=url,
                        model_name=model['name'], 
                        label_name=model['dataset'],
                        conf=client_setup['conf'] ,
                        nms=client_setup['nms'] ,
                        get_info=client_setup['info'] ,
                        client_timeout=client_setup['timeout']  )

    # 取得上傳的檔案
    uploaded_files = request.files.getlist("file[]")
    
    if mode == 'image':
        '''
        Image 讀取到檔案後直接轉成 Base64 的形式回傳到網頁上 ( 沒有進行存檔 )
        '''
        print_title("IMAGE MODE")
        data = []
        
        for file in uploaded_files:

            if not (file and allowed_file(file.filename)) : continue

            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)      # 轉換成 cv image
            
            filename = secure_filename(file.filename)       # 檢查檔案 & 儲存到自訂的目錄

            out_name = gen_uuid(filename)       # 取得獨特的UUID，避免 HTML 讀取舊的檔案
            out_pth = os.path.join(app.config['UPLOAD_FOLDER'], out_name)       # 取得除群位置
            
            # 進行 Inference
            results, image_draw =  client.image_infer(img, model['width'], model['height'], out_pth)

            info = parse_results(results)

            # 將資料記錄下來        
            data.append(  (out_name if isYolo else filename, info, cv2base64(image_draw)) )

        # 更新到 result 頁面
        return render_template('result.html', mode=mode, data=data, wid=len(data))
    
    elif mode == 'video':
        '''
        Video 不能透過file.read讀取圖片，所以只能用存取的方式
        '''
        # 如果非本地端則傳送過去儲存下來
        file = uploaded_files[0]
        filename = secure_filename(file.filename) 

        save_pth = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_pth)

        # 取得 UUID 的輸出
        out_name = gen_uuid(filename)       
        out_pth = os.path.join(app.config['UPLOAD_FOLDER'], out_name)

        # 進行推論取得結果
        ret, info = client.video_infer(save_pth, model['width'], model['height'], out_pth)

        

        return render_template('result.html', mode=mode, out_pth=out_name, info=info) 
    
    elif mode=='stream':
        '''
        Stream 每一幀讀取完就直接回傳
        '''
        global cap, isStream
        print_title("VIDEO MODE")

        # 如果非本地端則傳送過去儲存下來
        file = uploaded_files[0]
        filename = secure_filename(file.filename)        
        save_pth = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_pth)

        # Start Stream
        cap = cv2.VideoCapture(str(save_pth))
        # 允許串流
        isStream  = True
        
        return render_template('result.html', mode=mode)
            
    else:
        print_title("Error")
        sys.exit(1)

############################################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
