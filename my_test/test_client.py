import sys
import numpy as np
from paddle_serving_client import Client
from paddle_serving_app.reader import *
import cv2

preprocess = DetectionSequential([
    DetectionFile2Image(), 
    DetectionNormalize([102.9801,115.9465,122.7717],[1.0,1.0,1.0],False),
#    DetectionTranspose((2, 0, 1)), DetectionPadStride(32)
])

postprocess = RCNNPostprocess("label_list.txt", "output")
client = Client()

client.load_client_config("../serving_client/serving_client_conf.prototxt")
client.connect(['127.0.0.1:9292'])
print(client)
print("----------------------------------------------------")
im, im_info = preprocess(sys.argv[1])
fetch_map = client.predict(
    feed={
        "image": im,
        "im_shape": np.array([320,480,1]),
#	"im_shape": [[[480,640,3]]],
        "im_info": np.array([480,640,2]),
    },
    fetch=["multiclass_nms_0.tmp_0"],
    batch=True)
print(im_info)
print(im)
print(fetch_map)
print(im.shape)
print((np.array(list(im.shape[1:])+[1.0])).shape)
print(np.array(list(im.shape[1:])).reshape(-1).shape)
print(fetch_map)
print(sys.argv[1])
fetch_map["image"] = sys.argv[1]
postprocess(fetch_map)
