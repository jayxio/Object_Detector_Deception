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
from object_detectors.yolo_tiny_model_updated import YOLO_tiny_model_updated


class EOTB_attack:
    # init global variable
    model = None
    fromfile = None
    fromfolder = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = False
    filewrite_img = False
    filewrite_txt = False
    useEOT = True
    Do_you_want_ad_sticker = True
    disp_console = True
    weights_file = 'weights/YOLO_tiny.ckpt'
    # search step for a single attack
    steps = 300
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    w_img = 640
    h_img = 480

    def __init__(self, model, argvs = []):
        self.model = model
        self.success = 0
        self.overall_pics = 0
        self.argv_parser(argvs)
        self.build_model_attack_graph()
        
    def argv_parser(self,argvs, ):
        for i in range(1,len(argvs),2):
            # read picture file
            if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
            if argvs[i] == '-fromfolder' : self.fromfolder = argvs[i+1]
            if argvs[i] == '-frommaskfile' : self.frommaskfile = argvs[i+1]
            if argvs[i] == '-fromlogofile' : 
                self.fromlogofile = argvs[i+1]
            else:
                self.fromlogofile = None
                
            if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
            if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
            
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :self.imshow = True
                else : self.imshow = False
                    
            if argvs[i] == '-useEOT' :
                if argvs[i+1] == '1' :self.useEOT = True
                else : self.useEOT = False
                    
            if argvs[i] == '-Do_you_want_ad_sticker' :
                if argvs[i+1] == '1' :self.Do_you_want_ad_sticker = True
                else : self.Do_you_want_ad_sticker = False
                    
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False

    def build_model_attack_graph(self):
        if self.disp_console : print("Building YOLO attack graph...")


        # x is the image
        self.x = tf.placeholder('float32',[1,448,448,3])
        self.mask = tf.placeholder('float32',[1,448,448,3])

        self.punishment = tf.placeholder('float32',[1])
        self.smoothness_punishment=tf.placeholder('float32',[1])
        init_inter = tf.constant_initializer(0.001*np.random.random([1,448,448,3]))
        self.inter = tf.get_variable(name='inter',
                                     shape=[1,448,448,3],
                                     dtype=tf.float32,
                                     initializer=init_inter)

        # box constraints ensure self.x within(0,1)
        self.w = tf.atanh(self.x)
        # add mask
        self.masked_inter = tf.multiply(self.mask,self.inter)
        
        
        # compute the EOT-transformed masked inter in a batch, 
        if self.useEOT == True:
            print("Building EOT Model graph!")
            self.EOT_transforms = transformation.target_sample()
            num_of_EOT_transforms = len(self.EOT_transforms)
            print(f'EOT transform number: {num_of_EOT_transforms}')
            

            # broadcast self.masked_inter [1,448,448,3] into [num_of_EOT_transforms, 448, 448, 3]
            self.masked_inter_batch = self.masked_inter
            for i in range(num_of_EOT_transforms):
                if i == num_of_EOT_transforms-1: break
                self.masked_inter_batch = tf.concat([self.masked_inter_batch,self.masked_inter],0)

            # interpolation choices "NEAREST", "BILINEAR"
            self.masked_inter_batch = tf.contrib.image.transform(self.masked_inter_batch,
                                                                 self.EOT_transforms,
                                                                 interpolation='BILINEAR')

            
        else:
            self.masked_inter_batch = self.masked_inter
            print("EOT mode disabled!")

        
        # tf.add making self.w [1,448,448,3] broadcast into [num_of_EOT_transforms, 448, 448, 3]
        self.shuru = tf.add(self.w,self.masked_inter_batch)
        self.constrained = tf.tanh(self.shuru)
        
        # create session
        self.sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)

        init_dict = {'yolo_model_input': self.constrained,
                     'yolo_mode': "init_model",
                     'yolo_disp_console': self.disp_console,
                     'session': self.sess}

        # init a model instance
        self.object_detector = self.model(init_dict)
        self.C_target = self.object_detector.get_output_tensor()

        MODEL_variables = self.object_detector.get_yolo_variables()
        # Alternatives:
        # leave out tf.inter variable which is not part of yolo model
        # MODEL_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1:]
        # MODEL_variables = tf.contrib.framework.get_variables()[1:]

        # unused
        MODEL_variables_name = [variable.name for variable in MODEL_variables]


        # computer graph for norm 2 distance
        # init an ad example
        self.perturbation = self.x-self.constrained
        self.distance_L2 = tf.norm(self.perturbation, ord=2)
        self.punishment = tf.placeholder('float32',[1])

        # non-smoothness
        self.lala1 = self.masked_inter[0:-1,0:-1]
        self.lala2 = self.masked_inter[1:,1:]
        self.sub_lala1_2 = self.lala1-self.lala2
        self.non_smoothness = tf.norm(self.sub_lala1_2, ord=2)

        # loss is maxpooled confidence + distance_L2 + print smoothness
        self.loss = self.C_target+self.punishment*self.distance_L2+self.smoothness_punishment*self.non_smoothness

        # set optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-2)#GradientDescentOptimizerAdamOptimizer
        self.attackoperator = self.optimizer.minimize(self.loss,var_list=[self.inter])#,var_list=[self.adversary]

        # init and load weights by variables
        self.sess.run(tf.global_variables_initializer())

        # restore model variable
        saver = tf.train.Saver(MODEL_variables)
        saver.restore(self.sess,self.weights_file)


        if self.disp_console : print("Loading complete!" + '\n')

    def attack_from_file(self, filename, maskfilename, logo_filename=None):
        if self.disp_console : print('Detect from ' + filename)

        f = open(maskfilename)
        dic = xmltodict.parse(f.read())

        print("Input picture size:",dic['annotation']['size'])

        img = cv2.imread(filename)
        print(type(img),img.shape)
        mask = 0.000001*np.ones(shape=img.shape)

        # if there is logo file, prepare logo_mask
        if logo_filename is not None:
            logopic = cv2.imread(logo_filename)
            # flag indicates where the pixels' value will spared from ad perturbation
            flag = 100
            logo_mask = self.generate_logomask(logopic,
                                               flag)

        print("Generating Mask...")
        self.mask_list = dic['annotation']['object']
        resized_logo_mask_list = []
        for _object in self.mask_list:
            xmin = int(_object['bndbox']['xmin'])
            ymin = int(_object['bndbox']['ymin'])
            xmax = int(_object['bndbox']['xmax'])
            ymax = int(_object['bndbox']['ymax'])
            print(xmin,ymin,xmax,ymax)
            mask = self.generate_MaskArea(mask,
                                          xmin,
                                          ymin,
                                          xmax,
                                          ymax)

            # if there is logo file, draw logo where there is flags
            if logo_filename is not None:
                mask, resized_logo_mask = self.add_logomask(mask,
                                                            logo_mask,
                                                            xmin,
                                                            ymin,
                                                            xmax,
                                                            ymax)

                resized_logo_mask_list.append(resized_logo_mask)

        # usually the first area is what we need, when your make your mask you know
        self.attack_optimize(img, mask, logo_mask, resized_logo_mask_list[0])
    
    def attack_optimize(self, img, mask, logo_mask=None, resized_logo_mask=None):
        s = time.time()
        self.h_img,self.w_img,_ = img.shape
        
        if logo_mask is not None and resized_logo_mask is not None:
            img_resized_np = self.add_logo_on_input(img, logo_mask, resized_logo_mask)
        
        img_resized = cv2.resize(img, (448, 448))
        mask_resized = cv2.resize(mask, (448,448))
        
        img_resized_np = np.asarray(img_resized)
        inputs = np.zeros((1,448,448,3),dtype='float32')   # ni ye ke yi yong np.newaxis
        inputs_mask = np.zeros((1,448,448,3),dtype='float32')
        
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        
        inputs_mask[0] = mask_resized
        # image in numpy format
        self.inputs = inputs
        # hyperparameter to control two optimization objectives
        punishment = np.array([0.01])
        smoothness_punishment = np.array([0.5])


        # set original image and punishment
        in_dict = {self.x: inputs,
                   self.punishment: punishment,
                   self.mask: inputs_mask,
                   self.smoothness_punishment: smoothness_punishment}
        
        # set fetch list
        fetch_list = [self.object_detector.fc_19,
                      self.attackoperator,
                      self.constrained,
                      self.C_target,
                      self.loss]
        
        # attack
        print("YOLO attack...")
        for i in range(self.steps):
            # fetch something in self(tf.Variable)
            net_output = self.sess.run(fetch_list, feed_dict=in_dict)
            print("step:",i,"Confidence:",net_output[3],"Loss:",net_output[4][0])

            
        self.result = self.interpret_output(net_output[0][0])
        
        # reconstruct image from perturbation
        reconstruct_img_np_squeezed = self.save_np_as_jpg(net_output[2][0])
        
        print("Attack finished!")
        
        # choose to generate invisible clothe
        user_input = "Yes"
        while user_input!="No" and self.Do_you_want_ad_sticker is True:
            user_input = input("Do you want an invisible clothe? Yes/No:")
            if user_input=="Yes":
                print("Ok!")
                self.generate_sticker(reconstruct_img_np_squeezed, logo_mask, resized_logo_mask)
                break
            elif user_input=="No":
                print("Bye-Bye!")
                break
            else:
                print("Wrong command!")
                user_input = input("Do you want an invisible clothe? Yes/No:")
        
        self.show_results(img, self.result)
        
        strtime = str(time.time()-s)
        if self.disp_console : print('Elapsed time : ' + strtime + ' secs' + '\n')
            
    # save numpy pic as a jpg
    def save_np_as_jpg(self, x):
        '''
        x is numpy array between (-1,1)
        
        reconstruct_img_np_squeezed is numpy array between (0,1)
        '''
        # reconstruct image from perturbation
        ad_x=x
        ad_x_01=(ad_x/2.0)+0.5
        
        # bx.imshow only take value between 0 and 1
        squeezed=np.squeeze(ad_x_01)

        ad_x_squeezed=np.squeeze(ad_x)
        reconstruct_img_resized_np=(ad_x_squeezed+1.0)/2.0*255.0
        print("min and max in img(numpy form):",reconstruct_img_resized_np.min(),reconstruct_img_resized_np.max())

        reconstruct_img_np=cv2.resize(reconstruct_img_resized_np,(self.w_img,self.h_img))#reconstruct_img_BGR
        reconstruct_img_np_squeezed=np.squeeze(reconstruct_img_np)

        # write in sticker as jpg, idk how to write it in png. Help me!
        self.whole_pic_savedname=str(self.overall_pics)+".jpg" # time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".jpg"

        self.path = "./result/"
        
        is_saved=cv2.imwrite(self.path+self.whole_pic_savedname,reconstruct_img_np_squeezed)
        if is_saved:
            print("Result saved under: ",self.path+self.whole_pic_savedname)
        else:
            print("Saving error!")
        
        return reconstruct_img_np_squeezed
    
    # add logo on input
    def add_logo_on_input(self, pic_in_numpy_0_255, logo_mask=None, resized_logo_mask=None):
        is_saved = None
        
        _object = self.mask_list[0]
        xmin = int(_object['bndbox']['xmin'])
        ymin = int(_object['bndbox']['ymin'])
        xmax = int(_object['bndbox']['xmax'])
        ymax = int(_object['bndbox']['ymax'])
        print(xmin,ymin,xmax,ymax)
        
        if logo_mask is not None and resized_logo_mask is not None:
            ad_area_center_x = (xmin+xmax)/2
            ad_area_center_y = (ymin+ymax)/2

            # cv2.resize only eats integer
            resized_width = resized_logo_mask.shape[1]
            resized_height = resized_logo_mask.shape[0]

            paste_xmin = int(ad_area_center_x - resized_width/2)
            paste_ymin = int(ad_area_center_y - resized_height/2)
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height

            for i in range(paste_xmin,paste_xmax):
                for j in range(paste_ymin,paste_ymax):
                    if resized_logo_mask[j-paste_ymin,i-paste_xmin,0]==0.000001:
                        pic_in_numpy_0_255[j,i] = [255,255,255]

        
        sticker_in_numpy_0_255 = pic_in_numpy_0_255[ymin:ymax, xmin:xmax]
        
        assert sticker_in_numpy_0_255 is not None

        is_saved=cv2.imwrite('result/input.jpg',sticker_in_numpy_0_255)
        if is_saved:
            print("Sticker saved under:",'result/input.jpg')
        else:
            print("Sticker saving error")
        
        return pic_in_numpy_0_255
    
    # generate_sticker saved under result folder
    def generate_sticker(self, pic_in_numpy_0_255, logo_mask=None, resized_logo_mask=None):
        is_saved = None
        
        self.sitcker_savedname = "sticker_"+self.whole_pic_savedname
        _object = self.mask_list[0]
        xmin = int(_object['bndbox']['xmin'])
        ymin = int(_object['bndbox']['ymin'])
        xmax = int(_object['bndbox']['xmax'])
        ymax = int(_object['bndbox']['ymax'])
        print(xmin,ymin,xmax,ymax)
        
        sticker_in_numpy_0_255_original = pic_in_numpy_0_255[ymin:ymax, xmin:xmax]

        resize_ratio = logo_mask.shape[0]/resized_logo_mask.shape[0]
        
        new_sticker_width = int(sticker_in_numpy_0_255_original.shape[1] * resize_ratio)
        new_sticker_height = int(sticker_in_numpy_0_255_original.shape[0] * resize_ratio)
        new_sticker = cv2.resize(sticker_in_numpy_0_255_original,(new_sticker_width,new_sticker_height))
        
        if logo_mask is not None and resized_logo_mask is not None:
            ad_area_center_x = new_sticker_width/2
            ad_area_center_y = new_sticker_height/2

            # cv2.resize only eats integer
            resized_height = logo_mask.shape[0]
            resized_width = logo_mask.shape[1]

            paste_xmin = int(ad_area_center_x - resized_width/2)
            paste_ymin = int(ad_area_center_y - resized_height/2)
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height
            
            for i in range(paste_xmin,paste_xmax):
                for j in range(paste_ymin,paste_ymax):
                    if logo_mask[j-paste_ymin,i-paste_xmin,0]==0.000001:
                        new_sticker[j,i] = [255,255,255]
        
        assert new_sticker is not None
        is_saved=cv2.imwrite(self.path+'HD_'+self.sitcker_savedname,new_sticker)
        if is_saved:
            print("Sticker saved under:",str(self.path))
        else:
            print("Sticker saving error")

        return is_saved
    
    # generate mask
    def generate_MaskArea(self,
                          mask, 
                          xmin,
                          ymin,
                          xmax,
                          ymax):
        """
        make a mask where adversarial perturbed area
        """
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                for channel in range(3):
                    mask[j][i][channel] = 1

        return mask
    
    def generate_logomask(self,
                          logopic,
                          flag):
        """
        turn logopic into binary matrix logo_mask.
        """
        logo_mask = np.where(logopic>flag, 0.000001, 1)

        return logo_mask
    
    
    def add_logomask(self,
                     mask,
                     logo_mask,
                     xmin,
                     ymin,
                     xmax,
                     ymax):
        """
        put logo_mask onto the center of the ready-to-perturbed area. haha.
        """
        ad_width = xmax - xmin
        ad_height = ymax - ymin
        
        ad_area_center_x = (xmin+xmax)/2
        ad_area_center_y = (ymin+ymax)/2
        
        ad_ratio = ad_width/ad_height

        logo_height = logo_mask.shape[0]
        logo_width = logo_mask.shape[1]
        logo_ratio = logo_width/logo_height
        
        # make sure logo is contained by ad area
        if ad_ratio > logo_ratio:
            resize_ratio = ad_width/logo_height
        else:
            resize_ratio = ad_height/logo_width

        # cv2.resize only eats integer
        resized_height = int(logo_height*resize_ratio)
        resized_width = int(logo_width*resize_ratio)
        
        resized_logo_mask = None
        # skip cases where resize area is too small
        if resized_width!=0 and resized_height!=0:
            resized_logo_mask = cv2.resize(logo_mask, (resized_width,resized_height))
            
            paste_xmin = int(ad_area_center_x - resized_width/2)
            paste_ymin = int(ad_area_center_y - resized_height/2)
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height
            
            if (xmin+resized_height)<mask.shape[0] and (ymin+resized_width)<mask.shape[1]:
                mask[paste_ymin:paste_ymax,paste_xmin:paste_xmax] = resized_logo_mask
                # np.zeros([resized_height,resized_width,3])
            else:
                pass
        else:
            pass

        return mask, resized_logo_mask


    def interpret_output(self,output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
        debug_yan=np.zeros((7,7,2))
        for i in range(2):
            debug_yan[:,:,i]=np.multiply(class_probs[:,:,14],scales[:,:,i])
        #print(debug_yan.reshape(-1))
        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
        #print probs
        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]

        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]


        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def show_results(self,img,results):
        img_cp = img.copy()
        if self.filewrite_txt :
            ftxt = open(self.tofile_txt,'w')
        class_results_set = set()
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            class_results_set.add(results[i][0])
            if self.disp_console : print('    class : ' + 
                                         results[i][0] + ' , [x,y,w,h]=[' + 
                                         str(x) + ',' + str(y) + ',' + 
                                         str(int(results[i][3])) + ',' + 
                                         str(int(results[i][4]))+'], Confidence = ' + 
                                         str(results[i][5]))
                
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            if self.filewrite_txt :                
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if "person" not in class_results_set:
            self.success+=1
            print("Attack succeeded!")
        else:
            print("Attack failed!")
            
        if self.filewrite_img : 
            if self.disp_console : print('    image file writed : ' + self.tofile_img)
            cv2.imwrite(self.tofile_img,img_cp)  
            
        if self.imshow :
            cv2.imshow('YOLO_tiny detection',img_cp)
            cv2.waitKey(1)
            
        if self.filewrite_txt : 
            if self.disp_console : print('    txt file writed : ' + self.tofile_txt)
            ftxt.close()

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    def attack(self):
        if self.fromfile is not None and self.frommaskfile is not None:
            self.attack_from_file(self.fromfile, self.frommaskfile, self.fromlogofile)

        if self.fromfolder is not None:
            filename_list = os.listdir(self.fromfolder)
            # take pics name out and construct xml filename to read from
            for filename in filename_list:
                pic_name = re.match(r'\d+.JPG', filename)

                if pic_name is not None:
                    self.overall_pics+=1
                    print("Pics number:",self.overall_pics,"The",pic_name[0], "!")

                    pic_mask_name = pic_name[0][:-3]+"xml"
                    fromfile = self.fromfolder+"/"+pic_name[0]
                    frommask = self.fromfolder+"/"+pic_mask_name
                    
                    self.attack_from_file(fromfile, frommask)
                    
            print("Attack success rate:", self.success/self.overall_pics)
