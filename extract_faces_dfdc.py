import os
import json
import cv2
import glob

from extensions import VideoFaceExtractor

#input webm files directory
# input_dir = "C:\\Users\\paperspace\\Documents\\adelos\\dataset\\ffpp\\original_sequences\\youtube\\c40\\videos"

# #output mp4 files directory
# output_dir = "C:\\Users\\paperspace\\Documents\\adelos\\dataset\\real-fails-jpg"

# mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))

vfe = VideoFaceExtractor()
vfe.NB_FRAMES = 1

# for filename in mp4_files:
#     # get unique id from filename
#     base = os.path.basename(filename)
#     unique_id = os.path.splitext(base)[0]

#     # Extract faces
#     faces = vfe(filename)

#     # write faces to the output directory
#     for i, face in enumerate(faces):
#         output_img_path = os.path.join(output_dir, f"{unique_id}_{i}.jpg")
#         cv2.imwrite(output_img_path, face)

for idx in range(35):
    dfdc_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\dfdc\\dfdc_train_part_{idx}'
    target_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\train_faces'
    metadata = json.load(open(f"{dfdc_path}\\metadata.json"))
    for filename in metadata.keys():
        fileinfo = metadata[filename]
        label = 1 if fileinfo['label'] == 'FAKE' else 0
        base = os.path.basename(filename)
        unique_id = os.path.splitext(base)[0]
        try:
            faces = vfe(f"{dfdc_path}\\{filename}")
            for i, face in enumerate(faces):
                cv2.imwrite(f'{target_path}\\{label}\\{unique_id}_{i}.jpg', face)
        except:
            pass

for idx in range(35, 40):
    dfdc_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\dfdc\\dfdc_train_part_{idx}'
    target_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\test_faces'
    metadata = json.load(open(f"{dfdc_path}\\metadata.json"))
    for filename in metadata.keys():
        fileinfo = metadata[filename]
        label = 1 if fileinfo['label'] == 'FAKE' else 0
        base = os.path.basename(filename)
        unique_id = os.path.splitext(base)[0]
        try:
            faces = vfe(f"{dfdc_path}\\{filename}")
            for i, face in enumerate(faces):
                cv2.imwrite(f'{target_path}\\{label}\\{unique_id}_{i}.jpg', face)
        except:
            pass

for idx in range(40, 50):
    dfdc_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\dfdc\\dfdc_train_part_{idx}'
    target_path = f'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\val_faces'
    metadata = json.load(open(f"{dfdc_path}\\metadata.json"))
    for filename in metadata.keys():
        fileinfo = metadata[filename]
        label = 1 if fileinfo['label'] == 'FAKE' else 0
        base = os.path.basename(filename)
        unique_id = os.path.splitext(base)[0]
        try:
            faces = vfe(f"{dfdc_path}\\{filename}")
            for i, face in enumerate(faces):
                cv2.imwrite(f'{target_path}\\{label}\\{unique_id}_{i}.jpg', face)
        except:
            pass
