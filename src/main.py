import argparse
import cv2
from sys import platform
import math
import numpy as np
import random
import time
import logging as log
# import pyautogui

from input_feeder import InputFeeder
from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation

# from mouse_controller import MouseController

log.basicConfig(level=log.DEBUG)

BATCH_SIZE = 30

FACE_DETECTION_LOCATION = "../intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml"
FACIAL_LANDMARKS_DETECTION_LOCATION = "../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
HEAD_POSE_ESTIMATION_LOCATION = "../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
GAZE_ESTIMATION_LOCATION = "../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"

SCREEN_WIDTH = 1920 # pyautogui.size().width
SCREEN_HEIGHT = 1080 # pyautogui.size().height
SCREEN_X_LIMITS = [20, SCREEN_WIDTH-20]
SCREEN_Y_LIMITS = [20, SCREEN_HEIGHT-20]

_X = 0
_Y = 1
_H = 2
_W = 3

EYE_RADIUS = 20

MOUSE_PRECISION = 'medium'
MOUSE_SPEED = 'medium'

# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    log.error("Unsupported OS.")
    exit(1)

def draw_gaze_line(img, start_coord, end_coord):
    cv2.line(img, start_coord, end_coord, (0, 0, 255), 2)

def rescale(value, input_range, output_range):
    slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
    output = output_range[0] + slope * (value - input_range[0])

    return output

def get_linear_params(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def get_intersection_point(line1, line2):
    x1 = line1[0][_X]
    y1 = line1[0][_Y]
    x2 = line1[1][_X]
    y2 = line1[1][_Y]

    x3 = line2[0][_X]
    y3 = line2[0][_Y]
    x4 = line2[1][_X]
    y4 = line2[1][_Y]

    params1 = get_linear_params([x1, y1], [x2, y2])
    params2 = get_linear_params([x3, y3], [x4, y4])

    D  = params1[0] * params2[1] - params1[1] * params2[0]
    Dx = params1[2] * params2[1] - params1[1] * params2[2]
    Dy = params1[0] * params2[2] - params1[2] * params2[0]

    if D != 0:
        x = Dx / D
        y = Dy / D
        return (x, y)
    else:
        return False

def get_gaze_line(eye_center, xmin, ymin, gaze_vec_norm):
    start_coord = (eye_center[_X]+xmin,
                   eye_center[_Y]+ymin)
    end_coord = (eye_center[_X]+xmin+int((gaze_vec_norm[_X]+0.)*SCREEN_X_LIMITS[1]),
                 eye_center[_Y]+ymin-int((gaze_vec_norm[_Y]+0.)*SCREEN_Y_LIMITS[1]))

    return (start_coord, end_coord)

def get_mouse_point(gaze_mid_line, input_width, input_height):
    screen_right_line = [(SCREEN_X_LIMITS[1], SCREEN_Y_LIMITS[0]), (SCREEN_X_LIMITS[1], SCREEN_Y_LIMITS[1])]
    screen_left_line = [(SCREEN_X_LIMITS[0], SCREEN_Y_LIMITS[0]), (SCREEN_X_LIMITS[0], SCREEN_Y_LIMITS[1])]
    screen_bottom_line = [(SCREEN_X_LIMITS[0], SCREEN_Y_LIMITS[1]), (SCREEN_X_LIMITS[1], SCREEN_Y_LIMITS[1])]
    screen_top_line = [(SCREEN_X_LIMITS[0], SCREEN_Y_LIMITS[0]), (SCREEN_X_LIMITS[1], SCREEN_Y_LIMITS[0])]

    # rescale start point and end point from input frame size to screen size
    for i in range(2):
        rescaled_x = rescale(gaze_mid_line[i][_X], [0, input_width], SCREEN_X_LIMITS)
        rescaled_y = rescale(gaze_mid_line[i][_Y], [0, input_height], SCREEN_Y_LIMITS)
        gaze_mid_line[i] = (rescaled_x, rescaled_y)

    if gaze_mid_line[1][_X] > SCREEN_X_LIMITS[1]:
        gaze_mid_line[1] = get_intersection_point(screen_right_line, gaze_mid_line)

    if gaze_mid_line[1][_Y] > SCREEN_Y_LIMITS[1]:
        gaze_mid_line[1] = get_intersection_point(screen_bottom_line, gaze_mid_line)

    if gaze_mid_line[1][_X] < SCREEN_X_LIMITS[0]:
        gaze_mid_line[1] = get_intersection_point(screen_left_line, gaze_mid_line)

    if gaze_mid_line[1][_Y] < SCREEN_Y_LIMITS[0]:
        gaze_mid_line[1] = get_intersection_point(screen_top_line, gaze_mid_line)

    return (int(gaze_mid_line[1][_X]), int(gaze_mid_line[1][_Y]))

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Computer Pointer Controller")
    p_desc = "The location of the input file"
    t_desc = "Type of input, any of cam, video and image"
    d_desc = "The device name, if not 'CPU'"
    b_desc = "Draw bounding boxes"
    g_desc = "Draw gaze lines"

    # -- Add required and optional groups
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-p", "--input_path", help=p_desc, default="../bin/demo.mp4")
    optional.add_argument("-t", "--input_type", help=t_desc, default="video")
    optional.add_argument("-d", "--device", help=d_desc, default='CPU')
    optional.add_argument("-b", help=b_desc, action='store_true')
    optional.add_argument("-g", help=g_desc, action='store_true')
    args = parser.parse_args()

    return args

def infer_on_video(args):
    boundary_box_flag = args.b
    gaze_line_flag = args.g
    device = args.device
    input_path = args.input_path
    input_type = args.input_type

    # Initilize feeder
    feed = InputFeeder(input_type=input_type, input_file=input_path)
    feed.load_data()

    # Grab the shape of the input
    input_width = feed.getWidth()
    input_height = feed.getHeight()

    # Create a video writer for the output video
    out = cv2.VideoWriter('../out.mp4', CODEC, 30, (input_width,input_height))

    # mouse_controller = MouseController(MOUSE_PRECISION, MOUSE_SPEED)

    start_model_load_time=time.time()

    # model initialization
    face_detection = Face_Detection(FACE_DETECTION_LOCATION, device, extensions=CPU_EXTENSION)
    facial_landmarks_detection = Facial_Landmarks_Detection(FACIAL_LANDMARKS_DETECTION_LOCATION, device, extensions=CPU_EXTENSION)
    head_pose_estimation = Head_Pose_Estimation(HEAD_POSE_ESTIMATION_LOCATION, device, extensions=CPU_EXTENSION)
    gaze_estimation = Gaze_Estimation(GAZE_ESTIMATION_LOCATION, device, extensions=CPU_EXTENSION)

    total_model_load_time = time.time() - start_model_load_time

    counter = 0
    start_inference_time = time.time()

    # Process frames until the video ends, or process is exited
    for ret, batch in feed.next_batch(BATCH_SIZE):
        if not ret:
            break
        counter+=1
        gaze_lines = []
        out_frame = batch.copy()

        key = cv2.waitKey(60)

        # Face detection
        face_detection_output = face_detection.predict(batch)

        # face_detection_output = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        face_xmin = abs(int(face_detection_output[3] * input_width))
        face_ymin = abs(int(face_detection_output[4] * input_height))
        face_xmax = abs(int(face_detection_output[5] * input_width))
        face_ymax = abs(int(face_detection_output[6] * input_height))

        if (face_ymax-face_ymin) <= 0 or (face_xmax-face_xmin) <= 0:
            continue

        # Crop the face image
        face = batch[face_ymin:face_ymax, face_xmin:face_xmax]

        if boundary_box_flag == True:
            cv2.rectangle(out_frame, (face_xmin, face_ymin), (face_xmax, face_ymax), (255,255,0), 2)

        # Find facial landmarks (to find eyes)
        eyes = facial_landmarks_detection.predict(face)

        # Estimate head orientation (yaw=Y, pitch=X, role=Z)
        yaw, pitch, roll = head_pose_estimation.predict(face)

        eye_images = []
        for eye in eyes:
            face_height, face_width, _ = face.shape
            eye_xmin = int(eye[_X] * face_width - EYE_RADIUS)
            eye_ymin = int(eye[_Y] * face_height - EYE_RADIUS)
            eye_xmax = int(eye[_X] * face_width + EYE_RADIUS)
            eye_ymax = int(eye[_Y] * face_height + EYE_RADIUS)
            
            if (eye_ymax-eye_ymin) <= 0 or (eye_xmax-eye_xmin) <= 0:
                continue

            # crop and resize
            eye_images.append(face[eye_ymin:eye_ymax, eye_xmin:eye_xmax].copy())

            # Draw eye boundary boxes
            if boundary_box_flag == True:
                cv2.rectangle(out_frame,
                              (eye_xmin+face_xmin,eye_ymin+face_ymin),
                              (eye_xmax+face_xmin,eye_ymax+face_ymin),
                              (0,255,0),
                              2)

        # gaze estimation
        gaze_vec_norm = gaze_estimation.predict(eye_images, [yaw, pitch, 0])

        cos = math.cos(math.radians(roll))
        sin = math.sin(math.radians(roll))
        tmpx =  gaze_vec_norm[0]*cos + gaze_vec_norm[1]*sin
        tmpy = -gaze_vec_norm[0]*sin + gaze_vec_norm[1]*cos
        gaze_vec_norm = [tmpx, tmpy]

        # Store gaze line coordinations
        for eye in eyes:
            eye[_X] = int(eye[_X] * face_width)
            eye[_Y] = int(eye[_Y] * face_height)
            gaze_lines.append(get_gaze_line(eye, face_xmin, face_ymin, gaze_vec_norm))

        if gaze_line_flag:
            # Drawing gaze lines
            for gaze_line in gaze_lines:
                start_point = (gaze_line[0][_X], gaze_line[0][_Y])
                end_point = (gaze_line[1][_X], gaze_line[1][_Y])

                draw_gaze_line(out_frame, start_point, end_point)

        # start point of middle gaze line
        start_point = ((gaze_lines[0][0][_X]+gaze_lines[1][0][_X])/2, (gaze_lines[0][0][_Y]+gaze_lines[1][0][_Y])/2)

        # end point of middle gaze line
        end_point = ((gaze_lines[0][1][_X]+gaze_lines[1][1][_X])/2, (gaze_lines[0][1][_Y]+gaze_lines[1][1][_Y])/2)

        gaze_mid_line = [start_point, end_point]
        
        mouse_point = get_mouse_point(gaze_mid_line, input_width, input_height)

        log.debug("mouse_point[_X], mouse_point[_Y]: %s, %s", mouse_point[_X], mouse_point[_Y])

        cv2.circle(out_frame, mouse_point, 10, (255, 255, 255), -1)
        # mouse_controller.move(mouse_point[_X], mouse_point[_Y])

        # write out_frames with batch size
        for _ in range(BATCH_SIZE):
            # cv2.imshow("video", out_frame)
            out.write(out_frame)

        if key==27:
            break

    total_inference_time = time.time() - start_inference_time
    total_inference_time = round(total_inference_time, 1)
    log.info("total_inference_time: %s", total_inference_time)
    log.info("counter: %s", counter)
    fps = counter / total_inference_time

    with open('../stats.txt', 'w') as f:
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps)+'\n')
        f.write(str(total_model_load_time)+'\n')

    # Release the out writer, capture, and destroy any OpenCV windows
    log.info("Input stream ended...")
    # cv2.destroyAllWindows()
    out.release()
    feed.close()


def main():
    args = get_args()
    infer_on_video(args)

if __name__ == "__main__":
    main()
