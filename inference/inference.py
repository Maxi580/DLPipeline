from unet import unet_main
from yolo import yolo_main
from faster_rcnn import frcnn_main

from utils import *

MODEL_INFERENCE_INPUT = os.getenv('MODEL_INFERENCE_INPUT')

if __name__ == '__main__':
    """Each Function checks if corresponding Models exist and then uses them for Inference"""
    input_does_exist = check_directory_content([MODEL_INFERENCE_INPUT])

    if input_does_exist:
        yolo_main()
        frcnn_main()
        unet_main()
    else:
        print("No Input Data has been found.")

