from ..common.boundingbox import BoundingBox
import cv2
import numpy as np

""" 各自圖片進行撿均值（突顯該 圖像 重點） """
def subtract_avg(img):
    
    trg_img = img
    for i in range(3):
        avg=np.average(trg_img[:,:,i])
        trg_img[:,:,i]-=avg
    return trg_img

""" 整個數據集進行撿均值（突顯該 類別 重點） """
def subtract_offset(img, offset=( 103.939, 116.779, 123.68 )):
    trg_img = img
    for i in range(3):
        trg_img[:,:,i]=img[:,:,i]-offset[i]
    return trg_img

""" caffe mode of image processing """
def preprocess(img, input_shape, dtype=np.float32):
    
    img_resize = cv2.resize(img, (input_shape[1], input_shape[0])).astype(dtype)    
    img_avg = subtract_offset(img_resize)
    img_chw = img_avg.transpose( (2, 0, 1) ).astype(dtype)    
    return img_chw 

def postprocess(output, img_w, img_h, input_shape, conf_th=0.8, nms_threshold=0.5, letter_box=False):
    """
    if is objected detection in yolo ( with 200 TopK ):
        # TopK 是當初 TAO 訓練的時候給予的，代表每張圖最後最多輸出 TopK 個 BBOX
        (-1,200), (-1,200,4), (-1,200), (-1,200)
        num_detections: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
        nmsed_boxes: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
        nmsed_scores: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
        nmsed_classes: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes
    """
    # output = [ [num_of_detections], [bboxes], [scores], [labels] ]
    (detections, bboxes, scores, labels) = tuple([ np.squeeze(out) for out in output ]) 

    results= []
    for idx in range(detections):
        # print(scores[idx], '\t')
        x1, y1, x2, y2 = map(float, bboxes[idx]) # x1, y1, x2, y2
        x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
        if scores[idx]>=conf_th:
            results.append(BoundingBox(labels[idx], scores[idx], x1, x2, y1, y2, img_h, img_w))
    return results