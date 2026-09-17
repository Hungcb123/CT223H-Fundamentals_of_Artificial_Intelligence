from .src import dataset, evaluate, gradcam, models, multi_mag_fusion, stain_norm, train, transforms
import sys

# Ho tro import breakhis.dataset
sys.modules["breakhis.dataset"] = dataset
