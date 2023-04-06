import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt
import torch.nn as nn
import configparser

config = configparser.ConfigParser()  # 注意大小寫
config.read("label.ini")  # 配置檔案的路徑

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(17*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(-1, 17*2)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

class YOLOv7_pose_detection(object):
    def __init__(self):
        self.poseweights="yolov7-w6-pose.pt"
        self.poseweights2='min_loss_model.pt'
        self.device = select_device("0")
        self.model = attempt_load(self.poseweights, map_location=self.device)  # Load model
        self.model2 = MLP()
        self.model2.load_state_dict(torch.load(self.poseweights2))
        self.pose_check_count = 0

    @torch.no_grad()
    def detect(self,  frame , conf, iou,  mode=2):
        frame_count = 0  # count no of frames
        total_fps = 0  # count total fps
        time_list = []  # list to store time
        fps_list = []  # list to store fps
        person_loc_result = []
        person_class = []
        person_conf = []
        pose_check_alarm = False
        device = self.device  # select device
        model = self.model
        model2 = self.model2
        _ = model.eval()
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        frame_width = frame.shape[1]

        if mode == 2:
            model2.to(device)

        if frame is not None:  # if success is true, means frame exist
            orig_image = frame  # store frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))

            image = image.to(device)  # convert image data to device
            # image = image.float() #convert image to float precision (cpu)
            start_time = time.time()  # start time for fps calculation

            with torch.no_grad():  # get predictions
                output_data, _ = model(image)

            output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                  conf,  # Conf. Threshold.
                                                  iou,  # IoU Threshold.
                                                  nc=model.yaml['nc'],  # Number of classes.
                                                  nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                  kpt_label=True)

            output = output_to_keypoint(output_data)
            # print(output)

            im0 = image[0].permute(1, 2,
                                   0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
            im0 = im0.cpu().numpy().astype(np.uint8)

            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for i, pose in enumerate(output_data):  # detections per image

                if len(output_data):  # check if no pose
                    for c in pose[:, 5].unique():  # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        # print("No of Objects in Current Frame : {}".format(n))

                    for det_index, (*xyxy, conf, cls) in enumerate(
                            reversed(pose[:, :6])):  # loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        label = None if False else (
                            names[c] if False else f'{names[c]} {conf:.2f}')
                        point_loc, person_loc = plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                                                 line_thickness=3, kpt_label=True,
                                                                 kpts=kpts, steps=3,
                                                                 orig_shape=im0.shape[:2])
                        if mode == 1:
                            person_class.append(names[c])
                            person_conf.append(float(conf.cpu()))
                            person_loc_result.append(person_loc)


                        # 動作判定
                        if mode == 2:
                            point_loc = torch.FloatTensor(point_loc)
                            model2.eval()
                            with torch.no_grad():
                                prediction = model2(point_loc.to(device))  # 將tensor放到gpu上做predict
                            prediction = prediction.cpu()  # 因gpu上的tensor無法轉numpy格式, 所以要放回cpu上輸出
                            pred = prediction.detach().numpy()
                            print(pred)
                            if pred >= 0.88:  # threshold
                                pose_check_alarm = True
                                self.pose_check_count += 1
                                if self.pose_check_count >= 25:
                                    self.pose_check_count = 1000
                                    person_class.append(names[c])
                                    person_conf.append(float(conf.cpu()))
                                    person_loc_result.append(person_loc)
                                print("alarm!!!!!!!!!!")
                                cv2.putText(im0, "alarm!!!!!!!!!!", (120, 50), 0, 1, [0, 0, 255],
                                            thickness=2, lineType=cv2.LINE_AA)
                            else:
                                print("safe~")
                                cv2.putText(im0, "safe~", (10, 50), 0, 1, [225, 0, 0],
                                            thickness=2, lineType=cv2.LINE_AA)

            if pose_check_alarm is False:
                self.pose_check_count = self.pose_check_count//2


            return  person_class, person_conf, person_loc_result


    def human_detection_outputs_frame(self, img, classes, confidences, boxes):

        try:
            if len(classes) != 0:
                # print("check0")
                for classId, confidence, box in zip( classes, confidences, boxes):
                    # print("check1")
                    if classId == 'person' or True:
                        # print("check2")
                        confidence = round(confidence*100)
                        box = np.array(box)
                        x1y1 = (np.array([box[0], box[1]])).astype(int)
                        x2y2 = (np.array([box[0] + box[2], box[1] + box[3] - 10])).astype(int)
                        img = cv2.rectangle(img, tuple(x1y1), tuple(x2y2), (0, 255, 255), 1)
                        # img = cv2.putText(img, f'{self.names[classId]}:{confidence}', (x1y1[0], x1y1[1] - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        #                   (0, 255, 255), 2)
                        print(f"x1y1 : {x1y1}, x2y2 : {x2y2}")
                        img = cv2.putText(img, f'{classId}:{confidence}', (x1y1[0], x1y1[1] - 3),
                                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                          (0, 255, 255), 2)
        finally:

            return img


cap = cv2.VideoCapture(0)
yolov7_pose = YOLOv7_pose_detection()
while True:
    _, frame = cap.read()
    start = time.time()
    cls, conf, loc= yolov7_pose.detect(frame,conf=0.5,iou=0.6)
    end = time.time()
    frame = yolov7_pose.human_detection_outputs_frame( frame, cls, conf, loc)
    print(end-start)
    print(f'cls, conf, loc : {cls}, {conf}, {loc}')

    cv2.imshow('frame',frame)
    cv2.waitKey(1)










