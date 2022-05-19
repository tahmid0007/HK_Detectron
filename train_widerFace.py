# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# "D:/MS COCO Dataset/annotations/cocoface_instances_train2017.json"
# "D:/MS COCO Dataset/train2017/
# NUM_CLASSES = 2 # Person and Face
from detectron2.data.datasets import register_coco_instances
#register_coco_instances("a", {}, "./widerface/annotations/train_ann.json", "./widerface/WIDER_train/images")
#register_coco_instances("b", {}, "./widerface/annotations/val_ann.json", "./widerface/WIDER_val/images")
register_coco_instances("a", {}, "./HK/annotations/train_ann.json", "./HK/train")
register_coco_instances("b", {}, "./HK/annotations/train_ann.json", "./HK/train")
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.SEED = 42
cfg.INPUT.CROP.ENABLED = True
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("a",)
cfg.DATASETS.TEST = ("b")
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.MODEL.WEIGHTS = "" # frm scrath
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

#cfg.MODEL.RETINANET.NUM_CLASSES = 1
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
#cfg.MODEL.BACKBONE.FREEZE_AT = 4
#cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
#cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
#cfg.MODEL.RPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
cfg.INPUT.MIN_SIZE_TEST = 1000
cfg.INPUT.MIN_SIZE_TRAIN = 1000
cfg.INPUT.MAX_SIZE_TEST = 8000
cfg.INPUT.MAX_SIZE_TRAIN = 8000

cfg.TEST.DETECTIONS_PER_IMAGE = 200

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

def main():
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
def evalu(cfg):
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("b", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "b")
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

def show(cfg):
    im = cv2.imread("./test_4.jpg")
    
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("AI",out.get_image())
    cv2.imwrite("output/a.jpg", out.get_image())

if __name__ == '__main__':
    #main()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
    
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
    evalu(cfg)
    show(cfg)
    # os.system("tensorboard --logdir=output --host localhost --port 8088")