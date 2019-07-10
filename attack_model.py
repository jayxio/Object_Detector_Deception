import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import pdb
import xmltodict
import matplotlib.pyplot as plt
from PIL import Image
import time
import transformation
import os
import re

from object_detectors.yolo_tiny_model import YOLO_tiny_model
from attack_methods.gix_yolo_attack import GIX_yolo_attack
from attack_methods.xlab_yolo_attack import Xlab_yolo_attack


def main(argvs):
    attacker = GIX_yolo_attack(argvs)
    # attacker = Xlab_yolo_attack(argvs)
    attacker.attack()
    
if __name__=='__main__':    
    main(sys.argv)
