from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import keras_ocr

from gaugeReadingApi_v3 import GRApi
import csv

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="man2G.mp4",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def gaugeName(leftTop, frame, middle, iterName):
    x=leftTop[0]-middle[0]
    y=leftTop[1]-middle[1]
    if(x<0 and y<0): #left top
        n=iterName[0]
        iterName[0]+=1
        return '0_'+str(n).zfill(3)
    elif(x>0 and y<0): #right top
        n=iterName[1]
        iterName[1]+=1
        return '1_'+str(n).zfill(3)
    elif(x<0 and y>0): #left buttom
        n=iterName[2]
        iterName[2]+=1
        return '2_'+str(n).zfill(3)
    else: # right buttom
        n=iterName[3]
        iterName[3]+=1
        return '3_'+str(n).zfill(3)

def putTextonImg(gaugeVaule, leftTop, frame, middle):
    print(frame.shape)

    textOffsetX=-int(middle[0]/6)
    textOffsetY=-int(middle[1]/16)
    print(frame.shape)
    print('middle', middle)
    if(frame.shape[0]<=1080 and frame.shape[1]<=1920):
        middle=(frame.shape[1]-150,frame.shape[0])
    blackColor=(255,0,0)
    fontSize=4
    x=leftTop[0]-middle[0]
    y=leftTop[1]-middle[1]
    if(x<0 and y<0):
        cv2.rectangle(frame, (middle[0]+textOffsetX, middle[1]+textOffsetY*3), (middle[0], middle[1]), (255, 255,255), -1)
        #cv2.rectangle(frame, (middle[0]+textOffsetX-50, middle[1]+textOffsetY*3-50), (middle[0]+150, middle[1]), (255, 255,255), -1)
        cv2.putText(frame, gaugeVaule, (middle[0]+textOffsetX, middle[1]+textOffsetY), cv2.FONT_HERSHEY_SIMPLEX,fontSize, blackColor, 4, cv2.LINE_AA)
    elif(x>0 and y<0):
        cv2.rectangle(frame,(middle[0]*2+textOffsetX, middle[1]+textOffsetY*3), (middle[0]*2, middle[1]), (255, 255,255), -1)
        cv2.putText(frame, gaugeVaule, (middle[0]*2+textOffsetX, middle[1]+textOffsetY), cv2.FONT_HERSHEY_SIMPLEX,fontSize, blackColor, 4, cv2.LINE_AA)
    elif(x<0 and y>0):
        cv2.rectangle(frame, (middle[0]+textOffsetX, middle[1]*2+textOffsetY*3), (middle[0], middle[1]*2), (255, 255,255), -1)
        cv2.putText(frame, gaugeVaule, (middle[0]+textOffsetX, middle[1]*2+textOffsetY), cv2.FONT_HERSHEY_SIMPLEX,fontSize, blackColor, 4, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (middle[0]*2+textOffsetX, middle[1]*2+textOffsetY*3), (middle[0]*2, middle[1]*2), (255, 255,255), -1)
        cv2.putText(frame, gaugeVaule, (middle[0]*2+textOffsetX, middle[1]*2+textOffsetY), cv2.FONT_HERSHEY_SIMPLEX,fontSize, blackColor, 4, cv2.LINE_AA)
    return frame 

def recordGaugeDV(gaugeLT, middle):
    GDV=0
    if (gaugeLT[0]< middle[0] and gaugeLT[1]< middle[1]):
        GDV=0
    elif (gaugeLT[0]> middle[0] and gaugeLT[1]< middle[1]):
        GDV=1
    elif (gaugeLT[0]< middle[0] and gaugeLT[1]> middle[1]):
        GDV=2
    else:
        GDV=3
    return GDV


def video_capture(frame_queue, darknet_image_queue):
    #frame_end = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap.isOpened():
        print("影片3")
        ret, frame = cap.read()
        if not ret:
            break
        #frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #if frame_count == frame_end:
            #print("影片讀取完畢 3")
            #break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    frame_end = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap.isOpened():
        print("影片2")
        darknet_image = darknet_image_queue.get()
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_count == frame_end:
            print("影片讀取完畢 2")
            break
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        ####print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue, GVSL):


    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    aaa=0
    frame_end = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iterName = [0] * 4
    GV_num=None
    GV_num_C=0
    start100 = time.time()
    LLL=[]
    jj=0
    ff=0
    while cap.isOpened():
        print("影片1")
        frame = frame_queue.get()
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_count == frame_end:
            print("影片讀取完畢 1")
            for i in range(len(LLL)):
                print(LLL[i])
            break
        middle=(int(frame.shape[1]/2), int(frame.shape[0]/2))
        print('aaamiddleaaa',middle,aaa)
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []

        if frame is not None:
            difftime=-1
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                left, top, right, bottom=darknet.bbox2points(bbox_adjusted)
                framecopy=frame.copy()
                subFrame=frame[top:bottom,left:right]
                GDV=recordGaugeDV((left, top), middle)

                subFrameCopy=subFrame.copy()
                Gaugevalue=None
                if subFrame.size != 0:
                    gname=gaugeName((left,top), frame, middle, iterName)
                    Gaugevalue=GRApi(subFrame, gname,pipeline, GDV)
                    
                    if Gaugevalue==None :#or  Gaugevalue<0.6 or Gaugevalue>0.8:
                        Gaugevalue='Alarm'
                    frame=putTextonImg(str(Gaugevalue), (left,top), frame, middle)
                    
                jj+=1
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            imagecopy = darknet.draw_boxes(detections_adjusted, framecopy, class_colors)
   


            frame=cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
            cv2.imwrite('./mergeV/'+str(ff)+'.bmp', frame)
            ff+=1
            if not args.dont_show:
                #cv2.imshow('Inference', image)
                cv2.imshow('Inference frame', frame)
            if args.out_filename is not None:
                #video.write(image)
                video.write(frame)

            if cv2.waitKey(fps) == 27:
                break
            
            
        key = cv2.waitKey(25)
        if key == ord('n') or key == ord('p'):
            break
        aaa+=1
    cap.release()
    video.release()
    cv2.destroyAllWindows()


def testDialValue(img):
    #print(3333)
    #pipeline = keras_ocr.pipeline.Pipeline()
    print(4444)
    cv2.imwrite('aaa.png', img)
    
    prediction_groups = pipeline.recognize([img])
    print(5555)
    a=[]
    for i in range(len(prediction_groups[0])):
        a.append(prediction_groups[0][i][0])
    print(a)


def keras_ocr_set():
    global pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

if __name__ == '__main__':
    
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    #global pipeline
    #pipeline = keras_ocr.pipeline.Pipeline() #add 20220203
    
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keras_ocr_set()
    GVSL=[]
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, GVSL)).start()
