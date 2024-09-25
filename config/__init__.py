from .dataset_config import dataset_config
from .yowo_v2_config import yowo_v2_config


def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if 'yowo_v2_' in args.version:
        m_cfg = yowo_v2_config[args.version]

    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg


from environs import Env
import os
pwd = os.path.dirname(os.path.realpath(__file__))
from .config import *

env = Env()
env.read_env()

# minio-config
ENDPOINT = env.str("ENDPOINT", ENDPOINT)
ACCESS_KEY = env.str("ACCESS_KEY", ACCESS_KEY)
SECRET_KEY = env.str("SECRET_KEY", SECRET_KEY)
BUCKET = env.str("BUCKET", BUCKET)
SECURE = env.bool("SECURE", SECURE)

# model
WEIGHT = env.str("WEIGHT", WEIGHT)

# video
VIDEO_PATH = env.str("VIDEO_PATH", VIDEO_PATH)