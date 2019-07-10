# ODD(Object Detector Deception)

### 1. Introduction 

#### Purpose

Hello researchers and geeks,

This repo is a tool-box for attacking various object detectors with mainstream attack methods. For now, this repo only has a yolo model to play with. We welcome more models and attack methods to go onto the stage. Our aims is to provide players with easy access to neuron output manipulation and attack cost function customization. We welcome you to join us to explore various attack strategies.

Best wishes!

#### References:

[0]. Synthesizing Robust Adversarial Examples. https://arxiv.org/abs/1707.07397. Anish Athalye 2017.

[1]. Adversarial Patch. https://arxiv.org. TB Brown 2017.

[2]. http://gixnetwork.org/wp-content/uploads/2018/12/Building-Towards-Invisible-Cloak_compressed.pdf

[3]. The object detection code & paper is originally from : http://pjreddie.com/darknet/yolo/

[4]. The code we used here is a tensorflow implementation from : https://github.com/gliese581gg/YOLO_tensorflow

[5]. On Physical Adversarial Patches for Object Detection. https://arxiv.org/abs/1906.11897. Mark Lee, Zico Kolter 2019.

[6]. Fooling automated surveillance cameras: adversarial patches to attack person detection. https://arxiv.org/abs/1904.08653. Simen Thys, Wiebe Van Ranst, Toon Goedemé 2019.

[7]. Fooling Detection Alone is Not Enough: First Adversarial Attack against Multiple Object Tracking. arxiv.org/abs/1905.11026. Jia Y 2019.


### 2. Install

(1) git clone https://github.com/xuanwu-baidu/Object_Detector_Deception.git 

(2) Download weights file from https://drive.google.com/file/d/0B2JbaJSrWLpza0FtQlc3ejhMTTA/view?usp=sharing

(3) Put weights file under `./weights`

### 3. Usage

(1) Entering `./`

(2) Run command:
    `python YOLO_tiny_tf_attack.py -fromfile test/Darren.jpg -frommuskfile test/Darren.xml`

(3) Tuning the hyperparameter `self.punishment` and `attack steps` to control the optimization of the target.

(4) See source code under `./attack_methods/` for more attack option.

(5) When meeting the end condition, the program will save the adversary example will in `./result`.

### 4. Requirements

- TensorFlow
- Opencv3

### 5. Citation

If you use ODD for academic research, you are highly encouraged (though not required) to cite the following :

    @misc{ODD,
     author= {Somebody},
     title = {ODD: A Tool-box for Fooling Object Detectors},
     month = mar,
     year  = 2019,
     url   = {https://github.com/xuanwu-baidu/Object_Detector_Deception.git}
    }

### 6. Changelog

2019/07/08 : New Release!
