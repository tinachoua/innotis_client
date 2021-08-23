#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import cv2
import os

import subprocess as sp
import shlex

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels, WillLabels

# 為了讀取原生地檔案
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

import json

def print_title(text):
    # get terminal size
    columns, rows = os.get_terminal_size(0)
    print('-'*columns)
    print(text, '\n')

class Client:

    def __init__(self, url, model_name='yolov4', label_name='COCO', conf=0.9, nms=0.1, get_info=False, client_timeout=False):
        
        ##############################################
        # initialize
        self.url = url
        self.model = model_name
        self.label = self.get_label(label_name)

        self.confidence = conf
        self.nms = nms

        self.get_info = get_info
        self.client_timeout = client_timeout

        self.yolo = 'yolo' in model_name
        ##############################################
        # Create Triton Client
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        ##############################################
        # Check Health and Model Information
        self.check_triton_status()
        if self.get_info==True : self.get_model_info()

        ##############################################
        # get shape of model io
        self.input_name, self.output_name, self.input_dims, self.input_size, format, dtype = self.parse_model()
        c, h, w = self.input_size
        self.inputs_shape = [c, h, w] if self.input_dims==3 else [1, c, h, w] 

    def parse_model(self):
        '''
        解析模型資訊，取得輸入層 輸出層 資訊
        '''
        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

        metadata_request = service_pb2.ModelMetadataRequest( name=self.model, version="")
        model_metadata = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=self.model,version="")
        model_config = grpc_stub.ModelConfig(config_request).config

        if len(model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(len(model_metadata.outputs)))

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(len(model_config.input)))

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata.name + "' output type is " +
                            output_metadata.datatype)

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = (model_config.max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception("expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_metadata.shape)))

        if not self.yolo:
            if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
                (input_config.format != mc.ModelInput.FORMAT_NHWC)):
                raise Exception("unexpected input format " +
                                mc.ModelInput.Format.Name(input_config.format) +
                                ", expecting " +
                                mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                                " or " +
                                mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))
            if input_config.format == mc.ModelInput.FORMAT_NHWC:
                h = input_metadata.shape[1 if input_batch_dim else 0]
                w = input_metadata.shape[2 if input_batch_dim else 1]
                c = input_metadata.shape[3 if input_batch_dim else 2]
            else:
                c = input_metadata.shape[1 if input_batch_dim else 0]
                h = input_metadata.shape[2 if input_batch_dim else 1]
                w = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1]
            h = input_metadata.shape[2]
            w = input_metadata.shape[3]

        return (input_metadata.name, output_metadata.name, expected_input_dims, (c, h, w),
                input_config.format, input_metadata.datatype)

    def get_label(self, label_name):
        ''' 
        取得標籤物件 
        '''
        if label_name.upper()=='COCO':
            return COCOLabels
        elif label_name.upper()=='WILL':
            return WillLabels
        elif label_name.upper()=='IMAGENET':
            with open('imagenet1000.json') as f:
                label = json.load(f)
            return label

    def check_triton_status(self):
        '''
        檢查 Triton 的狀態
        '''
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)
        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        if not self.triton_client.is_model_ready(self.model):
            print("FAILED : is_model_ready")
            sys.exit(1)   

    def get_model_info(self):
        '''
        取得 Triton Server 的模型資訊
        '''
        try:
            metadata = self.triton_client.get_model_metadata(self.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = self.triton_client.get_model_config(self.model)
            if not (config.config.name == self.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)     

    def get_infer_stats(self):
        statistics = self.triton_client.get_inference_statistics(model_name=self.model)
        if len(statistics.model_stats) != 1:
            print("FAILED: get_inference_statistics")
            sys.exit(1)
        print(statistics)

    def get_video_info(self, cap):
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))             # 取得影像寬
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))            # 取得影像高
        fps = cap.get(cv2.CAP_PROP_FPS)                             # 取得FPS
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)                       # 設定   影片結尾
        legnth = round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 3)      # 取得時間郵戳 ( 毫秒 )
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)                       # 調整回 影片開頭
        
        return width,height,legnth,fps


    def render_dets(self, image, detected_objects):
        results = dict()
        for idx,box in enumerate(detected_objects):
            print(f"{self.label(box.classID).name}: {box.confidence}")
            results[idx]= [self.label(box.classID).name,  box.confidence]
            image = render_box(image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            size = get_text_size(image, f"{self.label(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            image = render_filled_box(image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            image = render_text(image, f"{self.label(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)
        
        if not results: results[0] = ['None', 0]

        return results, image
        

    def image_infer(self, image, width, height, output='', input_type='FP32'):

        print("Initialize ...")
        if image is None:
            print("FAILED: no input image")
            sys.exit(1)

        inputs, outputs = list(), list()
        
        inputs.append(grpcclient.InferInput(self.input_name, self.inputs_shape, input_type))
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        print("Creating buffer from image file...")
        image_buffer = preprocess(image, [width, height])   # image_buffer is for inference
        image_buffer = np.expand_dims(image_buffer, axis=0) if self.input_dims==4 else image_buffer
        inputs[0].set_data_from_numpy(image_buffer)    

        print("Invoking inference...")
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs)
        if self.get_info: 
            self.get_infer_stats()
        
        print("Inference Done !!!")

        print("Parsing Results...")
        result = results.as_numpy(self.output_name)
        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")

        if self.yolo:

            detected_objects = postprocess(result, image.shape[1], image.shape[0], [width, height], self.confidence, self.nms)
            print(f"Detected objects: {len(detected_objects)}")

            print("Rendering Bounding Box...")
            parsed_results, image_draw = self.render_dets(image.copy(), detected_objects)
            
            print("Output...")
            if output:
                cv2.imwrite(output, image_draw)
                print(f"Saved result to {output}")
            else:
                pass
                # cv2.imshow('image', image_draw)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            return parsed_results, image_draw
        # 如果不是 yolo 
        else:
            parsed_results = list()
            index = np.argmax(result)
            conf = result[index]
            classes = self.label[index]
            print_title('Result is : [{}] {}'.format(index, classes, conf))
            parsed_results.append( [index, classes, conf] )
            return parsed_results, image

    def video_infer(self, vid ,width , height, output, input_type='FP32'):
        print_title(output)
        res = dict()
        inputs, outputs = list(), list()
        
        inputs.append(grpcclient.InferInput(self.input_name, self.inputs_shape, input_type))
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        print("Opening input video stream...")
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"FAILED: cannot open video {vid}")
            sys.exit(1)
        
        print("Get Video Info")
        vwidth,vheight,vlegnth,vfps =self.get_video_info(cap)
        print_title('{}_{}_{}_{}'.format(vwidth,vheight,vlegnth,vfps))

        counter = 0
        if counter == 0 and output:        
            process = sp.Popen(shlex.split(f'ffmpeg -y -s {vwidth}x{vheight} -pixel_format bgr24 -f rawvideo -i pipe: -r {vfps} -vcodec libx264 -pix_fmt yuv420p -crf 24 {output}'), stdin=sp.PIPE)

        print("Invoking inference...")
        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                print("failed to fetch next frame")
                break

            # Data Preprocess
            print("Creating buffer from image file...")
            input_image_buffer = preprocess(frame, [width, height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0) if self.input_dims==4 else input_image_buffer
            inputs[0].set_data_from_numpy(input_image_buffer)

            ##############################################
            #Inference
            results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs)

            result = results.as_numpy(self.output_name)

            if self.yolo:

                detected_objects = postprocess(result, frame.shape[1], frame.shape[0], [width, height], self.confidence, self.nms)
                print(f"Frame {counter}: {len(detected_objects)} objects")
                counter += 1
                
                results, frame_draw = self.render_dets(frame.copy(), detected_objects)

                ########################################################################################################################################################################################
                # 輸出
                if output:
                    process.stdin.write(frame_draw.tobytes())
                else:
                    pass
                    # cv2.imshow('image', frame_draw)
                    # if cv2.waitKey(1) == ord('q'):  break
            else:
                print_title("not yolo")
                break

        cap.release()
        
        if output:
            process.stdin.close()
            process.wait()
            process.terminate()
            print_title("CLEAR")
        else:
            cv2.destroyAllWindows()
        
        print("All Done")
        return True

    def infer(self, mode, input, width, height, output):
        '''
        進行推論 可以選擇模式 : [ 'image', 'video' ]
        '''
        print_title(f'Running in {mode} mode')

        if mode == 'image':
            image = cv2.imread(str(input))
            self.image_infer(image, width, height, output)
        elif mode == 'video':
            self.video_infer(str(input), width, height, output)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov4',
                        help='Inference model name, default yolov4')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=608,
                        help='Inference model input width, default 608')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=608,
                        help='Inference model input height, default 608')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='',
                        help='Write output into file instead of displaying it')
    parser.add_argument('-c',
                        '--confidence',
                        type=float,
                        required=False,
                        default=0.8,
                        help='Confidence threshold for detected objects, default 0.8')
    parser.add_argument('-n',
                        '--nms',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Non-maximum suppression threshold for filtering raw boxes, default 0.5')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=24.0,
                        help='Video output fps, default 24.0 FPS')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')
    parser.add_argument('-l',
                        '--label',
                        type=str,
                        required=False,
                        default='COCO',
                        help='Taget Label')
    
    FLAGS = parser.parse_args()

    '''
    重新改寫之後 把 Client 寫成物件
    可以直接調用 infer
    這個將方便後續 Flask 的調用
    '''
    client = Client(url=FLAGS.url,
                    model_name=FLAGS.model, 
                    label_name=FLAGS.label, 
                    conf=FLAGS.confidence ,
                    nms=FLAGS.nms ,
                    info=FLAGS.model_info ,
                    client_timeout=FLAGS.client_timeout)

    client.infer(FLAGS.mode, FLAGS.input, FLAGS.width, FLAGS.height, FLAGS.out)