import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
from abc import *

def load_images_dict(input_folder):
    images_dict = {}
    for filename in os.listdir(input_folder):
        if "_mask" not in filename and "_predict" not in filename:
            name, ext = os.path.splitext(filename)
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            if img is not None:
                images_dict[name] = img
    return images_dict

def load_ground_truth_dict(input_folder):
    gt_dict = {}
    for filename in os.listdir(input_folder):
        if "_mask" in filename:
            name, ext = os.path.splitext(filename)
            key_name = name.replace("_mask", "")
            
            filepath = os.path.join(input_folder, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                gt_dict[key_name] = binary_mask
    return gt_dict

def calculate_average_f1_score(predicted_dict, gt_dict):
    f1_scores = []
    common_keys = set(predicted_dict.keys()).intersection(set(gt_dict.keys()))
    
    for key in common_keys:
        pred_mask = predicted_dict[key]
        gt_mask   = gt_dict[key]

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        pred_flat = (pred_mask.flatten() > 127).astype(np.uint8)
        gt_flat   = (gt_mask.flatten()   > 127).astype(np.uint8)

        score = f1_score(gt_flat, pred_flat)
        f1_scores.append(score)

    if len(f1_scores) == 0:
        return 0.0
    return np.mean(f1_scores)

input_folder = "./input"

images_dict = load_images_dict(input_folder)
print(f"Test set : {len(images_dict)} images")

gt_dict = load_ground_truth_dict(input_folder)
print(f"Segmentation mask : {len(gt_dict)} labels")

class SegmentationMapGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def predict_segmentation_map(self):
        pass

    @abstractmethod
    def predict_segmentation_maps(self):
        pass

#=========================<Your code starts here>========================

class HumanSegmentationMapGenerator(SegmentationMapGenerator):
    def __init__(self, images, out_folder):
        self.images = images
        self.out_folder = out_folder

    def predict_segmentation_map(self, img):
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        pred_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        return pred_mask
    
    def save_segmentation_map(self, mask, key):
        # This code is for visualization purposes only and is not actually needed
        save_path = os.path.join(self.out_folder, f"{key}_predict.png")
        cv2.imwrite(save_path, mask)

    def predict_segmentation_maps(self):
        predicted_dict = {}
        for key, img in self.images.items():
            pred_mask = self.predict_segmentation_map(img)
            predicted_dict[key] = pred_mask
        return predicted_dict

#=========================<Your code ends here>==========================

if __name__ == "__main__":
    generator = HumanSegmentationMapGenerator(images_dict, input_folder)
    predicted_dict = generator.predict_segmentation_maps()

    avg_f1 = calculate_average_f1_score(predicted_dict, gt_dict)
    print(f"Average F1 score: {avg_f1:.4f}")
