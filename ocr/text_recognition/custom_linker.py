import sys, os
import matplotlib.pyplot as plt

def get_images_sizes(crop_text_path):
    entries = os.listdir(crop_text_path)
    for entry in entries:
        entry = plt.imread(crop_text_path+ entry)
        print (entry.shape) 

def crop_bb_image(bounding_boxes, image, crop_text_path, file):
    for i, bounding_box in enumerate(bounding_boxes):
        coords = bounding_box.rstrip().split(",")
        coords = list(map(int, coords)) 
        x1 = min(coords[0], coords[2], coords[4], coords[6])
        y1 = min(coords[1], coords[3], coords[5], coords[7])
        x2 = max(coords[0], coords[2], coords[4], coords[6])
        y2 = max(coords[1], coords[3], coords[5], coords[7])
        if (y1 == y2 or x1 == x2):
            continue
        text = image[y1:y2, x1:x2]
        plt.imsave(crop_text_path+ str(i) + "_" + file, text)

def crop_images(images_path,text_detection_dir, crop_text_path):
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            image = plt.imread(images_path+ file)
            text_detection_file = ("res_"+file).split(".")[0] + ".txt"
            text_detection_file = os.path.join(text_detection_dir, text_detection_file)
            text_detection_file = open(text_detection_file, 'r') 
            bounding_boxes = text_detection_file.readlines() 
            crop_bb_image(bounding_boxes, image, crop_text_path, file)
            

crop_text_path = "demo_images_2/"
images_path = "../../data/"
text_detection_dir = "../CRAFT-pytorch/result"
# get_images_sizes(crop_text_path)
crop_images(images_path, text_detection_dir, crop_text_path)
