from detectron2.engine import  DefaultPredictor

import os
import pickle

from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# cfg_save_path = "OD_cfg.pickle" # for Object Detection
cfg_save_path = "IS_cfg.pickle" # for Instance Segmentation

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
# data = DatasetCatalog.get("my_dataset")

# image_path = "test/1.jpg"
videoPath = "test/Trim_01.mp4"

# on_image(image_path, predictor)
on_video(videoPath, predictor)