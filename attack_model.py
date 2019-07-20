'''
Author: Jay Xiong

'''

import sys

# choose attack method
from attack_methods.gix_yolo_attack import GIX_yolo_attack
from attack_methods.gix_yolo_attack_alpha import GIX_yolo_attack_alpha
from attack_methods.eotb_attack import EOTB_attack

# choose white-box models
from object_detectors.yolo_tiny_model import YOLO_tiny_model
from object_detectors.yolo_tiny_model_updated import YOLO_tiny_model_updated


def main(argvs):
    # choose attacker
    
    # attacker = GIX_yolo_attack(YOLO_tiny_model_updated, argvs)
    attacker = GIX_yolo_attack_alpha(YOLO_tiny_model_updated, argvs)
    # attacker = EOTB_attack(YOLO_tiny_model_updated, argvs)
    
    
    attacker.attack()
    
if __name__=='__main__':    
    main(sys.argv)
