import argparse
import threading
import cv2
import os
import time
import numpy as np
import torch
from copy import deepcopy
from PIL import Image

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection
from config import build_dataset_config, build_model_config
from models import build_model
from logs import setup_logger
from config import WEIGHT, VIDEO_PATH, DEBUG, BOT_TOKEN, MONGO_URI, ENDPOINT, ACCESS_KEY, SECRET_KEY, BUCKET, SECURE

from deep_sort.tracker import Tracker

from storage.s3_minio import S3Minio
from storage.mongo import MongoDBManager
from telegram.bot import MyBot
from utils import time_utils

# khoi tao doi tuong S3Minio
s3 = S3Minio(endpoint=ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, bucket=BUCKET, secure=SECURE)
# khoi tao doi tuong Mongo
mongo = MongoDBManager(uri=MONGO_URI)
# khoi tao doi tuong Tracker
tracker = Tracker()
# khoi tao doi tuong Telegram
myBot = MyBot(token=BOT_TOKEN)

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_path', default='test_model/outputs', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.8, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # class label config
    parser.add_argument('-d', '--dataset', default='trash',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False, 
                        help='show 14 action pose of AVA.')
    parser.add_argument('--num_classes', default=2, type= int)

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_medium', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=8, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    return parser.parse_args()

           

def multi_hot_vis(args, frame, out_bboxes, orig_w, orig_h, class_names, act_pose=False):
    # visualize detection results
    for bbox in out_bboxes:
        x1, y1, x2, y2 = bbox[:4]
        if act_pose:
            # only show 14 poses of AVA.
            cls_conf = bbox[5:5+14]
        else:
            # show all actions of AVA.
            cls_conf = bbox[5:]
    
        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        # score = obj * cls
        det_conf = float(bbox[4])
        cls_scores = np.sqrt(det_conf * cls_conf)

        indices = np.where(cls_scores > args.vis_thresh)
        scores = cls_scores[indices]
        indices = list(indices[0])
        scores = list(scores)

        if len(scores) > 0:
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1+3, y1+14+20*_))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 2)
                print(text[t])
    
    return frame

def save_storage(frame, logger_cam):
    current_timesteamp = time_utils.get_current_timestamp()
    date_timestamp = time_utils.get_midnight_timestamp_gmt7()
    camera = mongo.get_camera_by_rtsp(VIDEO_PATH)
    camera_id = str(camera['_id'])
    # save mongo
    new_detection = {
        "camera_id": camera_id,
        "detect_timestamp": current_timesteamp,
        "violation_date" : date_timestamp,
        "is_violation": False
    }
    violation_image_id = mongo.insert_violation(new_detection)
    violation_image_id = str(violation_image_id)
    logger_cam.info(f"New detected Littering id {violation_image_id} at {current_timesteamp}")
    cv2.imwrite(f'temp_file/{violation_image_id}.jpg', frame)
    s3.upload_file(f'temp_file/{violation_image_id}.jpg', f'{violation_image_id}.jpg')
    # noti telegram
    myBot.send_notification(f"New detected Littering at {str(time_utils.get_datetime())}", f'temp_file/{violation_image_id}.jpg')
    os.remove(f'temp_file/{violation_image_id}.jpg') 

def open_rtsp_camera(rtsp_url, logger_cam, retry_interval=5, max_retries=5, stop_cam = False):
    retries = 0

    while (retries < max_retries):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            print(f"Camera {rtsp_url} is opened.")
            logger_cam.info(f"Camera {rtsp_url} is opened.")
            return cap
        else:
            print(f"Error: Could not open camera {rtsp_url}. Retrying in {retry_interval} seconds...")
            logger_cam.error(f"Error: Could not open camera {rtsp_url}. Retrying in {retry_interval} seconds...")
            if stop_cam:
                retries += 1
            cap.release()
            time.sleep(retry_interval)

    print(f"Error: Could not open camera {rtsp_url} after multiple attempts. Exiting.")
    logger_cam.error(f"Error: Could not open camera {rtsp_url} after multiple attempts. Exiting.")
    # exit(1)
    return None

@torch.no_grad()
def detect(args, model, device, transform, class_names, class_colors):
    # path to save 
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # path to video
    rtsp_url = args.video
    cam_data = mongo.get_camera_by_rtsp(rtsp_url)

    name_log = f'ai_cam_{cam_data["_id"]}.log'
    logger_cam = setup_logger(name_log)
    # video
    cap = open_rtsp_camera(rtsp_url, logger_cam, stop_cam = False)
    if DEBUG:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (width, height)
        save_name = os.path.join(save_path, args.video.split('/')[-1].split('.')[0]+ '.avi')
        fps = 5
        out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # run
    video_clip = []
    image_list = []
    num_frame = 0
    # Tạo danh sách lưu các ID
    trash_ids = []
    while(True):
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {rtsp_url} failed to grab frame. Retrying...")
            logger_cam.error(f"Camera {rtsp_url} failed to grab frame. Retrying...")
            cap.release()
            cap = open_rtsp_camera(rtsp_url, logger_cam=logger_cam)
            continue
        if ret:
            # count frame
            num_frame += 1
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]

            # prepare
            if len(video_clip) < args.len_clip: # fix lỗi, khi nào đủ số frame mới cho vào model 
                frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))
                video_clip.append(frame_pil)
                continue
            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))
            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

            t0 = time.time()
            # inference
            outputs = model(x)
            print("inference time ", time.time() - t0, "s")

            # vis detection results
            if args.dataset in ['ava_v2.2']:
                batch_bboxes = outputs
                # batch size = 1
                bboxes = batch_bboxes[0]
                # multi hot
                frame = multi_hot_vis(
                    args=args,
                    frame=frame,
                    out_bboxes=bboxes,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    class_names=class_names,
                    act_pose=args.pose
                    )
            elif args.dataset in ['ucf24', 'trash']:
                batch_scores, batch_labels, batch_bboxes = outputs
                # batch size = 1
                scores = batch_scores[0]
                labels = batch_labels[0]
                bboxes = batch_bboxes[0]
                
                # rescale
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h])
                # one hot
                detections = [] # chi luu class Littering
                for i, bbox in enumerate(bboxes):
                    if scores[i] > args.vis_thresh:
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        label = int(labels[i])
                        if label == 0: # chi tinh class Littering
                            score = scores[i]
                            detections.append([x1, y1, x2, y2, score])
                if len(detections) > 0:
                    tracker.update(frame, detections)
                else:
                    continue
                for track in tracker.tracks:
                    box = track.bbox
                    left, top, right, bottom = box
                    track_id = track.track_id
                    frame_copy = deepcopy(frame)
                    frame_copy = cv2.rectangle(frame_copy, (int(left), int(top)), (int(right), int(bottom)), class_colors["Littering"], 2)
                    frame_copy = cv2.putText(frame_copy, f'id: {str(track_id)}', (int(left), int(top+20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, class_colors["Littering"], 1)
                    # Kiểm tra xem có ID nào mới không
                    if track_id not in trash_ids:
                        # Thêm ID mới vào danh sách
                        trash_ids.append(track_id)
                        # Lưu vào MongoDB
                        save_storage_thread = threading.Thread(target=save_storage, args=(frame_copy, logger_cam))
                        save_storage_thread.start()
                        
            if DEBUG:
                out.write(frame)
            # cv2.imwrite('debug.jpg', frame_resized)
            if args.gif:
                gif_resized = cv2.resize(frame, (200, 150))
                gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                image_list.append(gif_resized_rgb)

            if args.show:
                # show
                cv2.imshow('key-frame detection', frame)
                cv2.waitKey(1)

    # video.release()
    # out.release()

    # generate GIF
    # if args.gif:
    #     save_name = os.path.join(save_path, args.video.split('/')[-1].split('.')[0]+ '.gif')
    #     print('generating GIF ...')
    #     imageio.mimsave(save_name, image_list, fps=fps)
    #     print('GIF done: {}'.format(save_name))


if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()
    args.video = VIDEO_PATH
    os.makedirs('temp_file', exist_ok=True)
    cam_data = mongo.get_camera_by_rtsp(VIDEO_PATH)
    if cam_data is None:
        print(f"Không tìm thấy camera với RTSP URL: {VIDEO_PATH}")
        exit

    args.weight = WEIGHT

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']


    # class_colors = [(np.random.randint(255),
    #                  np.random.randint(255),
    #                  np.random.randint(255)) for _ in range(num_classes)]
    class_colors = {
        "Littering": [0,0,255],
        "Normal": [0,255,0]
        }
    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    detect(args=args,
        model=model,
        device=device,
        transform=basetransform,
        class_names=class_names,
        class_colors=class_colors)
