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
        print("Yolo Done!")
        frcnn_main()
        print("Frcnn Done!")
        unet_main()
        print("Unet Done!")
    else:
        print("No Input Data has been found.")

