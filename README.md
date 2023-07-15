# YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection
# Tiamo405
## Overview of YOWOv2
![image](./img_files/yowov2.png)
## 

## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n trash python=3.9
```

- Then, activate the environment:
```Shell
conda activate trash
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

## Visualization

![image](./img_files/trash_0001.gif)
![image](./img_files/Walking_0000.gif)

# Dataset
```
trash   __videos        __ Walking        __video1.mp4
        |               |                |__video2.mp4  
        |               |                |__.....
        |               |__ trashDumping  __video1.mp4
        |                                |__video2.mp4  
        |                                |__.....   
        |__label        __ Walking       __video1  __ frame11.txt
        |               |               |         |__ frame12.txt
        |               |               |         |__ .....        
        |               |               |__video2   __ frame11.txt
        |               |               |          |__ frame12.txt
        |               |               |          |__ .....
        |               |               |__ .....
        |               |__ trashDumping __video1 __ frame21.txt
        |                               |         |__ frame22.txt
        |                               |         |__ .....        
        |                               |__video2   __ frame21.txt
        |                               |          |__ frame22.txt
        |                               |          |__ .....
        |                               |__ .....    
        |__rgb-images    __ Walking     __video1   __ frame11.txt
        |               |               |         |__ frame12.txt
        |               |               |         |__ .....        
        |               |               |__video2   __ frame11.txt
        |               |               |          |__ frame12.txt
        |               |               |          |__ .....
        |               |               |__ .....
        |               |__ trashDumping __video1  __ frame21.txt
        |                                |         |__ frame22.txt
        |                                |         |__ .....        
        |                                |__video2   __ frame21.txt
        |                                |          |__ frame22.txt
        |                                |          |__ .....
        |                                |__ .....    
        |__testlist.txt
        |__trainlist.txt                
```
## Trash Dumping
You can download **Trash Dumping** from the following links:

[GDrive](https://github.com/tiamo405/Trash-Dumping)
## Custom data
* Create folder labels, rgb-images
```Shell
python extract_frame/video2frame.py --folder_videos trash/videos/Walking --label Walking
```
```Shell
python extract_frame/video2frame.py --folder_videos trash/videos/trashDumping --label trashDumping
```
* Create testlist.txt, trainlist.txt
```Shell
python trash/build_split.py
```

# Experiment

* Trash Dumping
  
|     Model      |   weight |
|----------------|----------|
|  YOWOv2-Nano   | [ckpt - chua co]() |
|  YOWOv2-Tiny   | [ckpt - chua co]() |
|  YOWOv2-medium | [ckpt](https://drive.google.com/file/d/1V6cENAtE7Tidxj1P01ixnIbzNVb6WtNf/view?usp=share_link) |


## Train YOWOv2
* Trash Dumping

For example:

```Shell
python train.py --cuda -d trash --root . -v yowo_v2_nano --num_workers 2 --num_classes 2 --eval_epoch 1 --max_epoch 8 --lr_epoch 2 3 4 5 -lr 0.0001 -ldr 0.5 -bs 8 -accu 16 -K 16
```

or you can just run the script:

```Shell
sh train_trash.sh
```

## Demo
```Shell
# run demo
python demo.py --cuda \
                -v yowo_v2_medium \
                --num_classes 2 \
                -size 224 \
                --weight checkpoints/trash/yowo_v2_medium/yowo_v2_medium_epoch_50.pth \
                --video video_test/v_Basketball_g01_c02.mp4 \
                --vis_thresh 0.5 \
                -d trash \
```
or you can just run the script:
```Shell
bash demo_trash.sh
```
## Note
```
Author  : Tran Phuong Nam
Contact : nam05052002@gmail.com
Mentor  : Cong Tran
Lab     : Naver
```


## References

[Github YOWO2](https://github.com/yjh0410/YOWOv2)
```
@article{yang2023yowov2,
  title={YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection},
  author={Yang, Jianhua and Kun, Dai},
  journal={arXiv preprint arXiv:2302.06848},
  year={2023}
}
```
