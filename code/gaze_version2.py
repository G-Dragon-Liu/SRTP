import os
import io
import sys
from string import Template
from pathlib import Path
import csv
import numpy as np
import PIL.Image as Image
import base64
import torch
import torchvision.transforms as transforms
import imageio
from aip import AipFace
import cv2
from gaze_tracking import GazeTracking

import datetime
from code.model import GazeLSTM
gaze = GazeTracking()

# from drive.gaze360.code.My_hdf5 import *
# import My_hdf5 首先测试不需要使用到My_hdf5
WIDTH, HEIGHT = 960, 720
total=set()

# Loading the model
def load_model(model_path, on_gpu):
    model_v = GazeLSTM()
    model = torch.nn.DataParallel(model_v)#.cuda()
    if not on_gpu:
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def cal_angle(v1, v2):
    x=np.array(v1)
    y=np.array(v2)
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle_hudu=np.arccos(cos_angle)
    angle_jiaodu=angle_hudu*360/2/np.pi
    gaze_object=False
    if angle_jiaodu<45:#小于45度就是注视
        gaze_object=True
    return angle_jiaodu,gaze_object

def get_file_content(file_path):
    """获取文件内容"""
    with open(file_path, 'rb') as fr:
        content = base64.b64encode(fr.read())
        return content.decode('utf8')


def face_score(file_path):
    """脸部识别分数"""
    result = a_face.detect(get_file_content(file_path), image_type, options)
    result1 = a_face.multiSearch(get_file_content(file_path), image_type,'srtp',options1)
    return result,result1

def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])
    return output


image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 配置百度aip参数
APP_ID = '23043036'
API_KEY = 'aI24WCSfPdZ38NGPjNgs9Hkb'
SECRET_KEY = 'F3PWDBB9EOeqEWshKmrrZc6HDiOpVg5K'
a_face = AipFace(APP_ID, API_KEY, SECRET_KEY)
image_type = 'BASE64'
#ptions = {'face_field': 'glasses',"max_face_num": 10}
options = {'face_field': 'glasses,age,gender,emotion', "max_face_num": 10}
options1 = {"max_face_num": 10,"match_threshold":0}
max_face_num = 10
#the variables to record the data
time=100
IDS_used = []
IDS=[]
date_last = []

with open('face_num.txt', 'r') as fw:
    unknown=int(fw.readlines()[0].rstrip('\n'))
fw.close()

video_capture = cv2.VideoCapture('testin1222.mp4') #打开摄像头 index默认自带摄像头0
#test1.mp4  test2.mp4  test3.mov
fps = video_capture.get(cv2.CAP_PROP_FPS )
fps = video_capture.get(cv2.CAP_PROP_FPS )
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH )),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#MPEG-4
Video_Writer = cv2.VideoWriter('1testout0301.mp4', fourcc, fps, size)   # 写出的文件

#fps = v_reader.get_meta_data()['fps']
out = imageio.get_writer('2testout0301.mp4')#fps=fps)
##############################################################################################################

GazeModel_DIR = 'code/' # 模型的位置
model_path = GazeModel_DIR+'gaze360_model.pth.tar'
gpu_ids = '-1'  # set to -1 if cpu; [0, 1, ...] if multiple gpus
my_device = torch.device('cuda:{}'.format(gpu_ids[0])) if (gpu_ids and gpu_ids != '-1') else torch.device('cpu')
on_gpu = True if (gpu_ids and gpu_ids != '-1') else False
'''
model = GazeLSTM()
model = torch.nn.DataParallel(model).cuda()
model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
'''
model = load_model(model_path,on_gpu)
model.eval()


color = (0, 0, 255)
font = cv2.FONT_HERSHEY_DUPLEX #字体、颜色
time=100

vector001=[0,0,1]
####################################主要的循环
# 初始化
# 需要一个dict存储所有的人脸，face location + id
tracking_id = dict()  # 存储第i帧的所有id及其face_location
identity_last = dict()  # 存储id 对应的人脸location + eyes
identity_next = dict()
final_results = dict()  # 保存bbox
# known_face_encodings=dict()# 由id索引到人脸编码,唯一
N = 7  # 25 取前面7帧
i = 0
j = 0
known_faces = 0
trans = transforms.ToTensor()
face_names = []
ready = 0
id_num = 0  # 记录人脸数
ith = 0  # 表示第i帧
id_frame_number = {}  # 单个id被捕捉到的帧数
###
FILE_DIR = 'unknown/'
di = 0

f = open('test.csv','w',encoding='utf-8')
writer = csv.writer(f)
writer.writerow(["a","b","c"])

while (True):  # video_capture.isOpened()):
    # frame= frame[:, :, ::-1]
    # ret, frame = cap.read()  # 获取

    ret, frame = video_capture.read()  # 获取
    print(ready)
    if ret == False:
        break
    di += 1
    process = False

    image = frame.copy()
    image = cv2.resize(image, (WIDTH, HEIGHT))
    #image = image.astype(float)
    # out.append_data(image)

    process = False
    if ready % 50 <= 10:  # 每50次获得一次人脸框 ，这里不占时间，占时间的是后面的计算
        process = True
        cv2.imwrite('tem.jpg', frame)
        file_path = 'tem.jpg'
        result, result1 = face_score(file_path)
        # print(result)
        # print(result1)
        # print(ready)
        tracking_id = dict()  # 处理一次需要清空，防止占用内存
        identity_last = dict()
        ith = 0
    ready += 1
    if (result['error_code']) == 0 and process == True:  # 均衡,每30帧处理5次 5/30
        know_face = 0
        face = 0
        for item, item1 in zip(result['result']['face_list'], result1['result']['face_list']):
            face += 1
            x = int(item['location']['left'])
            y = int(item['location']['top'])
            w = item['location']['width']
            h = item['location']['height']

            centerx = x + w / 2
            centery = y + h / 2

            x1 = int(max(centerx - 0.7 * w, 1))
            x2 = int(min(centerx + 0.7 * w, len(frame[0]) - 2))
            y1 = int(max(centery - 1.1 * h, 1))
            y2 = int(min(centery + 0.7 * h, len(frame) - 2))

            age = item['age']
            emotion = item['emotion']['type']
            prob = item['emotion']['probability']
            gender = item['gender']['type']
            name = item1['user_list'][0]['user_id']
            name_score = item1['user_list'][0]['score']

            if gender == 'female':
                gender_num = 0
            else:
                gender_num = 1

            if name_score < 60:
                name = 'unknown'
            else:
                know_face += 1
                id = int(name[2:])
                IDS.append(id)  # 已经识别的用户，这一次扫描出现了。在数据库中应该有对应的记录

            if (emotion == 'happy' or emotion == 'surprise'):
                emotion_num = 1
                intere = 1

            elif emotion == 'angry' or emotion == 'sad' or emotion == 'disgust' or emotion == 'fear':
                emotion_num = -1
                intere = 1
            else:
                emotion_num = 0
                intere = 0

            if name_score < 60 and ready % 30 == 1:
                unknown += 1
                face_im = frame[y1:y2, x1:x2]
                new_UID = 'u_' + str(unknown)
                face_path = FILE_DIR + new_UID + '.jpg'
                cv2.imwrite(face_path, face_im)
                image2 = get_file_content(face_path)
                a_face.addUser(image2, image_type, 'srtp', new_UID)
                IDS.append(unknown)  # 新的用户，这一次扫描出现了，那么放入IDS中。
                # total.add(new_UID)
            bbox_head = (x1 - 0.6 * w, y1 - 0.6 * h, x2 + 0.5 * w, y2 + 0.2 * h)  # 给眼动

            rate_w = WIDTH / frame.shape[1]
            rate_h = HEIGHT / frame.shape[0]
            x1 = int(x1 * rate_w)
            x2 = int(x2 * rate_w)
            y1 = int(y1 * rate_h)
            y2 = int(y2 * rate_h)

            total.add(name)
            dy = 0
            if y2 > 720: continue
            if y2 > 360: dy = -360
            if x2 > 960: continue

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.rectangle(image, (int(bbox_head[0] * rate_w), int(bbox_head[1] * rate_h)),
                          (int(bbox_head[2] * rate_w), int(bbox_head[3] * rate_h)), (255, 0, 0), 0)
            cv2.putText(image, 'emoj:%s' % emotion, (x1, y2 + dy + 34), font, 0.7, color, 1)
            if bbox_head is None: continue
            eyes = [(bbox_head[0] + bbox_head[2]) / 2.0, (0.65 * bbox_head[1] + 0.35 * bbox_head[3])]
            identity_next[0] = (bbox_head, eyes)  # UID 的位置以及eyes 的信息

            bbox_head = list(bbox_head)
            if bbox_head[0] <= 0: bbox_head[0] = 0
            if bbox_head[2] >= WIDTH: bbox_head[2] = WIDTH
            if bbox_head[1] <= 0: bbox_head[1] = 0
            if bbox_head[3] >= HEIGHT: bbox_head[3] = HEIGHT



            if (item['glasses']['type'] == 'none'):
                if (x1 >= 10):

                    theframe = frame[int(bbox_head[1]):int(bbox_head[3]), int(bbox_head[0]):int(bbox_head[2])]

                    gaze.refresh(theframe)
                    theframe = gaze.annotated_frame()
                    text = ""

                    if gaze.is_blinking():
                        text = "Blinking"
                        cv2.putText(image, text, (x1, y2 + dy + 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)

                    h_ratio = ratio = gaze.horizontal_ratio()
                    v_ratio = ratio = gaze.vertical_ratio()

                    if h_ratio is None: continue
                    mygaze = (h_ratio, v_ratio)
                    mygaze = eyes = np.asarray(mygaze).astype(float)

                    bbox, eyes = identity_next[0]
                    eyes = np.asarray(eyes).astype(float)
                    eyes[0], eyes[1] = eyes[0] / float(frame.shape[1]), eyes[1] / float(frame.shape[0])

                    pt1 = (int(eyes[0] * WIDTH), int(eyes[1] * HEIGHT))
                    arrow_size = 200
                    pt2 = (int(eyes[0] * WIDTH + arrow_size * (mygaze[0] - 0.7)),
                           int(eyes[1] * HEIGHT + arrow_size * (mygaze[1] - 0.6)))
                    cv2.arrowedLine(image, pt1, pt2, (0, 0, 255), 1)
                    # myangle, is_gaze = cal_angle([-mygaze[0], mygaze[1], -gaze[2]], vector001)  # x和z的翻转是坐标系决定的 is_gaze表示注视
                    cv2.putText(image, "h ratio: " + str(h_ratio), (x1, y2 + dy + 64), font, 0.5, color, 1)
                    cv2.putText(image, "h ratio: " + str(v_ratio), (x1, y2 + dy + 64 + 15), font, 0.5, color, 1)


                    writer.writerow(mygaze)
            else:

                eyes = [(bbox_head[0] + bbox_head[2]) / 2.0, (0.65 * bbox_head[1] + 0.35 * bbox_head[3])]
                identity_next[0] = (bbox_head, eyes)  # UID 的位置以及eyes 的信息

                input_image = torch.zeros(7, 3, 224, 224)
                count = 0
                for j in range(0, N):  # 一张用7遍
                    new_im = Image.fromarray(frame, 'RGB')
                    bbox, eyes = identity_next[0]
                    new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    input_image[count, :, :, :] = image_normalize(
                        transforms.ToTensor()(transforms.Resize((224, 224))(new_im)))
                    count = count + 1
                input_image = input_image.view(1, N * 3, 224, 224)  # 3*N
                bbox, eyes = identity_next[0]
                bbox = np.asarray(bbox).astype(int)
                # output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224).cuda())
                output_gaze, _ = model(input_image.view(1, 7, 3, 224, 224))
                mygaze = spherical2cartesial(output_gaze).detach().numpy()
                eyes = np.asarray(eyes).astype(float)
                eyes[0], eyes[1] = eyes[0] / float(frame.shape[1]), eyes[1] / float(frame.shape[0])
                mygaze = mygaze.reshape((-1))

                pt1 = (int(eyes[0] * WIDTH), int(eyes[1] * HEIGHT))
                arrow_size = 500
                pt2 = (int(eyes[0] * WIDTH - mygaze[0] * arrow_size), int(eyes[1] * HEIGHT - mygaze[1] * arrow_size))
                cv2.arrowedLine(image, pt1, pt2, (0, 0, 255), 1)
                myangle, is_gaze = cal_angle([-mygaze[0],mygaze[1], -mygaze[2]], vector001)  # x和z的翻转是坐标系决定的 is_gaze表示注视
                cv2.putText(image, 'angle:' + str(myangle), (x1, y2 + dy + 64), font, 0.3, color, 1)
                cv2.putText(image, 'gaze?:' + str(is_gaze), (x1, y2 + dy + 64 + 15), font, 0.3, color, 1)
                cv2.putText(image, 'gaze_at:x y z ' + str(-mygaze[0]) + ' ' + str(mygaze[1]) + ' ' + str(-mygaze[2]),
                            (x1, y2 + dy + 64 + 30), font, 0.3, color, 1)
                writer.writerow(mygaze)

    total.discard('unknown')
    cv2.imshow('gaze', image)
    out.append_data(image)
    image = np.uint8(image)

    Video_Writer.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.close()
        f.close()
        video_capture.release()
        Video_Writer.release()
        break


# video_capture.release()
# Video_Writer.release()
# out.close()
cv2.destroyAllWindows()
with open('face_num.txt', 'w') as fw:
    fw.writelines([str(unknown) + '\n'])

fw.close()

