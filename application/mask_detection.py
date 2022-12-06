#单类别单目标检测模板。wmingzhu
import io
import os
from flask import Flask,request
import cv2
import json
import base64
import torch
import numpy as np
import datetime
from PIL import Image
from public_logger import logger
from utils.tools import plot_one_box,letterbox,non_max_suppression,scale_coords

datetime_clear = lambda :datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","").replace("-","").replace(":","")
app = Flask(__name__)

class Detector(object):
    def __init__(self,model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')#yolov5原train保存的模型
        self.model = torch.load(model, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        self.confthre = 0.6
        self.nmsthre = 0.45
        self.img_size = 640


    def inference(self, image,save_path="./results/"):
        img = letterbox(image,new_shape=self.img_size)[0]#接收的是单张图片
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR到RGB;hwc到chw
        img = np.ascontiguousarray(img)#转为元素内存连续数组，提高数据处理速度
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()#uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred,self.confthre,self.nmsthre,classes=None,agnostic=False)#NMS筛选的结果是(1,n,6)，并且按conf从大到小排列
        pred = pred[0][:1]#只要第一个,即conf最高的
        pred[:,:4] = scale_coords(img.shape[2:],pred[:,:4],image.shape)#将bbox缩放回原图
        pred = [round(v) for v in list(np.array(pred.cpu())[0])[:4]]#得到整数坐标[x1,y1,x2,y2]
        # plot_one_box(pred,image,color=(0,0,255))#标框
        # cv2.imwrite(save_path+"{}.jpeg".format(len(os.listdir(save_path))),image)
        return pred

detector = Detector("./best.pt")

@app.route('/get_bbox',methods=['POST'])
def predict():
    data = json.loads(request.data)
    try:
        data = base64.b64decode(data["image"])
    except Exception as e:
        logger.error("b64图片解码错误:{}".format(e))
        return "b64 decode error"
    try:
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        image = image[:, :, ::-1]  # RGB2BGR
    except Exception as e2:
        logger.error("b64图片格式错误:{}".format(e2))
        return "b64 image format error"

    try:
        prediction = detector.inference(image)
        return json.dumps(prediction)
    except Exception as e3:
        logger.error("推理出错:{}".format(e3))
        return "inference error"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

