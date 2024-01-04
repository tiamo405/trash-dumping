import cv2
import numpy as np
import os

def vis_targets(video_clips, targets):
    """
        video_clips: (Tensor) -> [B, C, T, H, W]
        targets: List[Dict] -> [{'boxes': (Tensor) [N, 4],
                                 'labels': (Tensor) [N,]}, 
                                 ...],
    """
    batch_size = len(video_clips)

    for batch_index in range(batch_size):
        video_clip = video_clips[batch_index]
        target = targets[batch_index]

        key_frame = video_clip[:, :, -1, :, :]
        tgt_bboxes = target['boxes']
        tgt_labels = target['labels']

        key_frame = convert_tensor_to_cv2img(key_frame)
        width, height = key_frame.shape[:-1]

        for box, label in zip(tgt_bboxes, tgt_labels):
            x1, y1, x2, y2 = box
            label = int(label)

            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height

            # draw bbox
            cv2.rectangle(key_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0), 2)
        cv2.imshow('groundtruth', key_frame)
        cv2.waitKey(0)


def convert_tensor_to_cv2img(img_tensor):
    """ convert torch.Tensor to cv2 image """
    # to numpy
    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
    # to cv2 img Mat
    cv2_img = img_tensor.astype(np.uint8)
    # to BGR
    cv2_img = cv2_img.copy()[..., (2, 1, 0)]

    return cv2_img


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.3):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        # cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 + 20)), 0, text_scale, cls_color)

    return img

def vis_detection(frame, scores, labels, bboxes, vis_thresh, class_names, class_colors, name_video, num_frame):
    ts = 1
    os.makedirs(os.path.join('test_model/outputs', name_video), exist_ok= True)
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            label = int(labels[i])

            cls_color = class_colors[class_names[label]]

            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[label], scores[i])

                print(mess)
            else:
                cls_color = [255, 0, 0]
                mess = None
                # visualize bbox
            frame = plot_bbox_labels(frame, bbox, mess, cls_color, text_scale=ts)

            if class_names[label] =='trashDumping':
                os.makedirs(os.path.join('test_model/outputs', name_video, 'trash'), exist_ok= True)
                cv2.imwrite(os.path.join('test_model/outputs', name_video, 'trash', str(num_frame).zfill(5)+'.jpg'), frame)

    # save all frame
    frame = cv2.putText(frame, f'frame: {str(num_frame)}', (50,50), 1, 1, (0,255,0))
    os.makedirs(os.path.join('test_model/outputs', name_video, 'all'), exist_ok= True)
    cv2.imwrite(os.path.join('test_model/outputs', name_video, 'all', str(num_frame).zfill(5)+'.jpg'), frame)

    return frame
        