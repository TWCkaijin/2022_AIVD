# 


import json
import os
import time

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

import cv2
import threading
import requests
import time


MODEL_PATH = 'AIGO TFLite'
VIDEO_NUMBER = 0
#VIDEO_NUMBER = 'test_video.mp4'
#VIDEO_NUMBER = 'http://192.168.0.14:5000/live'

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
LEFT_KEY = 2424832
RIGHT_KEY = 2555904
UP_KEY = 2490368
DOWN_KEY = 2621440

Data_Base = [["2102202001","統一陽光豆漿",380,0],
             ["1902202002","黑松FIN補給飲料",220,0],
             ["2202202001","黑松沙士",280,0],
             ["2202201001","黑松茶花綠茶",260,0],
             ["2002202005","麥香綠茶",160,0],
             ["2002202007","麥香紅茶",140,0],
             ["2002202009","麥香奶茶",200,0],
             ["1701902001","統一肉燥麵",190,0]]


def get_prediction(image, interpreter, signature):
    # process image to be compatible with the model
    input_data = process_image(image, image_shape)

    # set the input to run
    interpreter.set_tensor(model_index, input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}

    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)


def Read_Key_Data(crop_img,image_src,scale,box_left,box_top, \
                  key_detect,key_flag,display_key,confirm,concentration,location,shift_times,LOD):
    read_dir_key = cv2.waitKeyEx(1)
    #print(image_src.shape[1],image_src.shape[0], scale,LOD)
    if ((box_left + scale  >=  image_src.shape[1])and(box_top + scale >= image_src.shape[0])):
        key_flag =  0
        box_top = 0
        box_left = 0
        confirm = 1
        location.append(0) 
        if (shift_times >= 2):  
            if (location[0] == 0):
                scale = scale*2
        
                concentration = concentration + 1
            elif(location[0] != 0):
                scale = scale/(2**concentration)
                concentration = 0 
        else:
            scale = scale/(2**concentration)
            concentration = 0

    if ((box_left + scale <=  image_src.shape[1])and(box_top + scale > image_src.shape[0])):
        box_top = 0
        box_left = 0
        scale = scale/(2**concentration)
        concentration = 0 
        
    if ( box_left + scale >= image_src.shape[1] ):
        #display_key=cv2.waitKey(1000)
        key_flag = 0
        box_left = 0
        box_top = box_top + LOD
        confirm = 1    
        
    if  (key_flag == 1)and( box_top + scale  <= image_src.shape[0] ):
        #display_key=cv2.waitKey(1000)
        key_flag = 0
        box_left = box_left + LOD
        confirm = 1
        shift_times = shift_times + 1 

          
    if ((read_dir_key == 113) or (read_dir_key == 27)):
        key_detect = 1
        print("QUIT")
        
    return crop_img,image_src,scale,box_left,box_top, \
           key_detect,key_flag,display_key,confirm,concentration,location,shift_times,LOD


def Show_ROI_Info(image_src,display_key,frame_width,frame_height,scale,box_left,box_top,Info_Text):
    cv2.putText(
        image_src, 
        "Enlarge: "+str(concentration),
        (int(0.85 * frame_width), int(0.97 * frame_height)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,255,255), 1, cv2.LINE_AA)
    
    cv2.putText(
        image_src, 
        Info_Text,
        (int(box_left + 5),int(box_top + 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,255,255), 1, cv2.LINE_AA)
        

def Track_func(detail,image_src,box_top,box_left,scale,locate_num,t,should_track,Base_time):
    tracker = cv2.TrackerCSRT_create()  # 追蹤器建置
    area = (int(box_left) , int(box_top) , int(scale) , int(scale) ) #(X1,Y1,X2,Y2)
    tracker.init(frame, area)    # 初始化
    str3 = "碳足跡:"+str(detail[1][2])+"g"
    t_Line = threading.Thread(target = send_Line , args=("碳足跡紀錄系統", str(detail[1][1]) , str3 ))
    t_Line.start()
    check_time = time.time()
    print(("商品"+detail[1][1], "碳足跡:"+str(detail[1][2])+"g" ," "  ))
    print("sending msg")
    while (should_track )and(check_time - Base_time<= 4):
        check_time = time.time()
        success, point = tracker.update(frame)   # 左上、右下座標回傳   
        p1 = [int(point[0]), int(point[1])]
        p2 = [int(point[0] + scale), int(point[1] + scale)]
        cv2.rectangle(frame, p1, p2, (0,0,255), 3)   # 框住追蹤物
        cv2.putText(frame,detail[1][0],(p1[0]+10,p1[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,255,255), 1, cv2.LINE_AA)
        n_key = cv2.waitKey(1)

def send_Line(str1,str2,str3):
    LINE_event_name = 'thing'
    LINE_key = 'lrft_gvwD9wuQ3PJYwSGdvkX9ehtai4RQwkvT0-s_O9'
    # Your IFTTT LINE_URL with event name, key and json parameters (values)
    LINE_URL='https://maker.ifttt.com/trigger/' + LINE_event_name + '/with/key/' + LINE_key

    ###
    ###r = requests.post(LINE_URL, params={"value1":str1, "value2":str2,"value3":str3})
    ###
def TP (image_src):
    global frame   
    frame = image_src
    return frame


def main():    
    global signature_inputs
    global input_details
    global model_inputs
    global signature_outputs
    global output_details
    global model_outputs
    global image_shape
    global model_index
    global scale
    global location
    global concentration
    global shift_times
    global LOD
    global CF_total 
 
   
    
    with open( MODEL_PATH + "/signature.json", "r") as f:
        signature = json.load(f)

    model_file = signature.get("filename")

    interpreter = tflite.Interpreter(MODEL_PATH + '/' + model_file)
    interpreter.allocate_tensors()

    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}
    image_shape = model_inputs.get("Image").get("shape")
    model_index = model_inputs.get("Image").get("index")

    cap = cv2.VideoCapture(VIDEO_NUMBER)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    #����
    key_detect = 0
    display_key = 0 #��J������

    #��m�ؤo
    scale=50 #�j�p
    box_left = 0
    box_top = 0
    
    #���ѫY��
    key_flag = 0 #�O�_��w(1�S���A0��)
    classification_flag = 0 #���Ѩ쪫��
    confirm = 1 #�T�{��������ʧ@(1�����A0������)  
    frame_counter = 0
    
    #�����Y��
    scale  = 160        #int(input("type in your scale"))
    LOD =160                 #int(input('type in your level of detail\nThe smaller the more meticulous'))
    location = []
    concentration = 0
    shift_times = 0
    locate_num = 0
    t =[]
    should_track = False
    i_token = 0


    
   
    

    
    

    while (key_detect == 0) and (cap.isOpened()):

        if (type(VIDEO_NUMBER) == int) or ('http' in VIDEO_NUMBER):
            ret,image_src = cap.read()
            k = TP(image_src)
            
        else:
            ret,image_src = cap.read()
            frame_counter += 1
            if frame_counter == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                frame_counter = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                
        if (ret != True):
            break

        frame_width = image_src.shape[1]
        frame_height = image_src.shape[0]

        if (confirm == 0):
            crop_img = image_src
        if (confirm == 1):
            temp_img = image_src[int(box_top):int(box_top + scale),int(box_left):int(box_left + scale)]
            crop_img = cv2.resize(temp_img,
                (224,224),interpolation=cv2.INTER_CUBIC)

        image = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

        if  ((key_flag == 0) and (confirm == 1)):
            prediction = get_prediction(image, interpreter, signature)
            Label_name = signature['classes']['Label'][prediction['Confidences'].index(max(prediction['Confidences']))]
            Info_Text = Label_name + " " + str(round(max(prediction['Confidences']),3))
            classification_flag = 1
            
        
         
            if (str(Label_name) != "0"):
                shift_time = time.time()
                

                for i in range(len(Data_Base)):
                    try :
                        _object = Data_Base[i].index(Label_name)
                        if  (shift_time - Data_Base[i][3] >= 3 ):
                            locate_num = locate_num + 1
                            i_token = i
                            Data_Base[i][3] = time.time()
                            location.append([locate_num-1,Data_Base[i],(box_top,box_left),(box_top + scale,box_left + scale)])
                            should_track = True
                                      
                           
                    except: 
                        #print("Label not found")
                        None
                        
            if should_track :
                t.append(threading.Thread(target = Track_func,args =(location[locate_num-1], \
                                                                        image_src,box_top,box_left,scale,locate_num-1, \
                                                                        t,should_track,Data_Base[i_token][3])))
                should_track = False
                t[locate_num-1].start()
                
        
       
                    
                
        if ((key_flag == 0) and ( classification_flag == 1)):
            key_flag = 1
            classification_flag = 0
            confirm = 0
            
        crop_img,image_src,scale,box_left,box_top,key_detect, \
        key_flag,display_key,confirm,concentration,location,shift_times,LOD = Read_Key_Data( \
        crop_img,image_src,scale,box_left,box_top,key_detect, \
        key_flag,display_key,confirm,concentration,location,shift_times,LOD)
    
        Show_ROI_Info(image_src,display_key,frame_width,frame_height,scale,box_left,box_top,Info_Text)
        
        
        if (key_flag == 1):
            cv2.rectangle(image_src,
                (int(box_left),int(box_top)),(int(box_left+scale),int(box_top+scale)),
                (0,0,225),2)
            cv2.imshow('Detecting ....',image_src)

        elif (key_flag == 0):
            cv2.rectangle(image_src,
                (int(box_left),int(box_top)),(int(box_left+scale),int(box_top+scale)),
                (0,225,0),2)
            cv2.imshow('Detecting ....',image_src)

         

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




            
        
   
    

           
   
   