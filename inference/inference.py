from unet import unet_main
from yolo import yolo_main
from faster_rcnn import frcnn_main



if __name__ == '__main__':
    """Each Function checks if corresponding Models exist and then uses them for Inference"""
    yolo_main()
    frcnn_main()
    unet_main()
