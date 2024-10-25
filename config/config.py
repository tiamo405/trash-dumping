# minio-config
ENDPOINT ="192.168.5.106:9000"
ACCESS_KEY="minioadmin"
SECRET_KEY="minioadmin"
BUCKET="trash-dumping"
SECURE = False

# model
# WEIGHT = 'checkpoints/ava/yowo_v2_large_ava_k32.pth'
WEIGHT = 'checkpoints/trash/yowo_v2_medium/yowo_v2_medium_epoch_50.pth'

# video
# VIDEO_PATH = 'test_model/video_test/2.mp4' # or camera
VIDEO_PATH = 'rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701'

# debug
DEBUG = True

# mongo
MONGO_URI = "mongodb+srv://nam05052002:vWAESJh9bivKcBYY@cluster0.27uon.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
