import os
import cv2
import glob

from extensions import VideoFaceExtractor

#input webm files directory
input_dir = "D:\\mp4"

#output mp4 files directory
output_dir = "D:\\jpg"

mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))

vfe = VideoFaceExtractor()

for filename in mp4_files:
    # get unique id from filename
    base = os.path.basename(filename)
    unique_id = os.path.splitext(base)[0]

    # Extract faces
    faces = vfe(filename)

    # write faces to the output directory
    for i, face in enumerate(faces):
        output_img_path = os.path.join(output_dir, f"{unique_id}_{i}.jpg")
        cv2.imwrite(output_img_path, face)
