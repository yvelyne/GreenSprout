# Reference: https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/

import argparse
import os
import sys
from cv2 import cv2
import numpy as np
import os
import time
import CarModule
import mapmatching
import pymssql
from sklearn.externals import joblib

def initmodel(classes_path, cfg_path, weights_path):
    """
    初始化模型
    :param classes_path:
    :param cfg_path:
    :param weights_path:
    :return:
    """
    # Load class names needed to detect
    with open(classes_path) as f:
        class_names = f.read().strip().split("\n")

    # Read Yolov3 models from model configuration and trained data
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return class_names, net


def getOutputsNames(net):
    """
    获得模型的输出层
    :param net:
    :return:
    """
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    """
    去除低概率的预测值
    :param frame:
    :param outs:
    :return:
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    class_ids = []
    confidences = []
    boxes = []
    

    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 过滤预测概率低于概率阈值的预测，此处的阈值不等同于conf_threshold
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                #过滤太小的框
                #if detection[2]<0.05 or detection[3]<0.05:
                #    continue
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    # 使用非极大值抑制去除包含概率低的预测
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshod, nms_threshod)
    return indices,boxes,class_ids,frame.shape

def carProcessing(indices,boxes,class_ids,frame_size,car_list0,count_car,LaneParam,vanishing_point):   
    #初始化车辆匹配列表。行：本张图片的一辆车，与上一张图片所有车的匹配分数
    matching_scores=[]
    frame_width=frame_size[1]
    frame_height=frame_size[0]
    #本张图的车辆列表
    car_list=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        matching_s=[]
        #if class_ids[i]!=2 and class_ids[i]!=5:#这个物体不是车
        #    continue
        if left<0 or top<0 or left+width>frame_width or top+height>frame_height:
            continue
        car=CarModule.CarClass(frame[top:top+height,left:left+width],top,left,width,height,frame,LaneParam,vanishing_point)
        #car=CarModule.CarClass(frame[top:top+height,left:left+width],top,left,width,height)
        car_list.append(car)            
        #有上一张图片   
        if len(car_list0)!=0 :
            #与前一张图的每一个车辆比对，计算分数
            for previous_car in car_list0:          
                #s=car.size_matching(previous_car)     
                matching_s.append(car.cos_matching(previous_car))
                #matching_s.append(car.SURFmatch(previous_car))
                #除了直方图，还要加上位置和大小。并且考虑一对多的问题。去掉太小的车。（识别不准确、位置不准确
            matching_scores.append(matching_s)
        else:#没有上一张图片
            count_car+=1
            car_num=count_car
            car.set_car_num(car_num)
            drawPredCar(car)
    if len(car_list0)==0 or len(car_list)==0:
        return car_list,count_car
    #开始匹配
    imax=np.argmax(matching_scores,axis=0) #每一列的最大值。前一张图每辆车在这张图的匹配。行：本张 列：前一张
    #遍历前图车辆
    for j in range(len(car_list0)):
        for i in range(len(car_list)):
            if i==imax[j] and matching_scores[i][j]>0.85: #是前图某车的匹配，且分数>0.85
                car_num=car_list0[j].car_num
                car_list[i].set_car_num(car_num)
                car_list[i].set_delta_distance(car_list0[j].dy)
    #处理未匹配到的，并画图
    for car in car_list:
        if car.car_num==-1: #未在前图找到匹配
            count_car+=1
            car_num=count_car
            car.set_car_num(car_num)            
        drawPredCar(car)

    return car_list,count_car
def drawPredCar(car):
    """
    根据预测结果在图片上画出矩形框
    """
    if car.lane_num==1:
        rectcolor=(0,255,0) #绿
    elif car.lane_num==2:
        rectcolor=(255,0,0) #蓝    
    elif car.lane_num==3:
        rectcolor=(0,255,255) #黄
    elif car.lane_num==4:
        rectcolor=(255,0,255) 
    else: #car.lane_num==-1: #错误
        rectcolor=(0,0,255) 
    
    cv2.rectangle(frame, (car.left, car.top), (car.left+car.width, car.top + car.height), rectcolor, 3)
    #label = "{0}: {1:.4f},count={2}".format(class_names[class_id], confidence,count_car)
    label = "{0},y={1:.1f},x={2:.1f}".format(car.car_num,car.dy,car.dx)
    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    #top = max(car.top, labelSize[1])
    cv2.putText(frame, label, (int(car.x_center), car.top + car.height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def drawPred(class_id, confidence, left, top, right, bottom,car_num):
    """
    根据预测结果在图片上画出矩形框
    :param class_id:
    :param confidence:
    :param left:
    :param top:
    :param right:
    :param bottom:
    :return:
    """
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    #label = "{0}: {1:.4f},count={2}".format(class_names[class_id], confidence,count_car)
    label = "{0}".format(car_num)
    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


if __name__ == '__main__':
    
    # Constant Parameters
    conf_threshod = 0.25  # 包含物体的概率阈值
    nms_threshod = 0.4  # 非极大值抑制阈值
    input_width = 416  # 模型输入图片宽度
    input_height = 416  # 模型输入图片高度

    # Files Paths
    root=r'd:\darknet-master\\'
    classes_path = root+"data/images/car.names"#r"data\coco.names"
    cfg_path = root+ "cfg/yolov3-tiny-car.cfg"# r"cfg\yolov3.cfg"
    weights_path = root+ "cfg/yolov3-tiny-car_10000.weights" #r"cfg\yolov3.weights"
 
    inputdir=r"D:\BaiduNetdiskDownload\20190429\pic\original"
    outputdir=r"D:\BaiduNetdiskDownload\20190429\pic\yolo-tiny"
    # Class Labels
    class_names = []
    
    # Model
    net = None
    fnames=os.listdir(inputdir)

    start=time.time()
    #初始化car_list
    car_list0=[]
    #初始化车辆计数
    count_car=0
    #测试用 车道线初值
    LastLaneParam=np.array([[-7.91408430e-01,  1.37122453e+03],[ 2.66310987e+00, -4.89425724e+02]])
    LastVanishingPoint=np.array([944, 600])
    
    #连接路网数据库
    conn=pymssql.connect(server='127.0.0.1',database='SPDB')
    #高架分类器
    svc=joblib.load(r"C:\Users\Lenovo\Desktop\lvmiao\py\gaojia\all_svc_train_model.m")

    for fname in fnames:
        inputpath=inputdir+"\\"+fname
        outputpath=outputdir+"\\"+fname
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group()
        parser.add_argument("-input", "--input_path", help="输入路径",default=inputpath)
        parser.add_argument("-output", "--output_path", help="输出文件夹",default=outputpath)

        args = parser.parse_args()

        cap = cv2.VideoCapture(args.input_path)

        class_names, net = initmodel(classes_path, cfg_path, weights_path)

        #window_name = "Yolov3 detection"
        #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(window_name, 1200, 800)
        ret, frame = cap.read()

        if ret:#not ret
            #print("Done processing!")
            print("Output file is stored as", args.output_path)
            #cv2.waitKey(0)
            #break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        indices,boxes,class_ids,frame.shape=postprocess(frame, outs)

        #加载车道线参数
        VanishingPoint,LaneParam=CarModule.get_lane_param(fname)       
        if VanishingPoint[0]==-1: #检测失败，用上次的有效结果
            LaneParam=LastLaneParam
            VanishingPoint=LastVanishingPoint
        else:           
            LastLaneParam=LaneParam #保存有效结果
            LastVanishingPoint=VanishingPoint
        #路网匹配
        rid,rname=mapmatching.Imap_matching(inputdir,fname,frame,[2,3,5,2,1],conn,svc)
        #在图上标注
        cv2.putText(frame, f"{rid},{rname}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
        
        #处理车辆。计算距离、车速和所在车道并画图
        car_list,count_car=carProcessing(indices,boxes,class_ids,frame.shape,car_list0,count_car,LaneParam,VanishingPoint)
        car_list0=car_list
        if count_car>100:
            count_car=0
                    
        cv2.imwrite(args.output_path, frame.astype(np.uint8))

        #cv2.imshow(window_name, frame)
    end=time.time()-start
    print(end/len(fnames))
    #cv2.destroyAllWindows()

    """   
    # Command Line Options
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group()
    group.add_argument("-i", "--image", help="检测图片", action="store_true")
    group.add_argument("-v", "--video", help="检测视频", action="store_true")
    parser.add_argument("-input", "--input_path", help="输入路径")
    parser.add_argument("-output", "--output_path", help="输出文件夹")

    args = parser.parse_args()

    if args.image:
        if not os.path.isfile(args.input_path):
            print("Input image file ", args.input_path, " doesn't exit")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input_path)
    elif args.video:
        if not os.path.isfile(args.video):
            print("Input video file ", args.input_path, " doesn't exit")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input_path)
    else:
        # 打开本地摄像头
        cap = cv2.VideoCapture(0)

    if not args.image:
        vid_writer = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                     (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    class_names, net = initmodel(classes_path, cfg_path, weights_path)

    window_name = "Yolov3 detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    while cv2.waitKey(1) < 0:

        ret, frame = cap.read()

        if not ret:
            print("Done processing!")
            print("Output file is stored as", args.output_path)
            cv2.waitKey(0)
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        postprocess(frame, outs)

        if args.image:
            cv2.imwrite(args.output_path, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv2.imshow(window_name, frame)

    cv2.destroyAllWindows()
    """


