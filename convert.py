import os
import cv2
import glob

#input webm files directory
input_dir = "C:\\Users\\paperspace\\Documents\\adelos\\dataset\\real-webm"

#output mp4 files directory
output_dir = "C:\\Users\\paperspace\\Documents\\adelos\\dataset\\real-mp4"

webm_files = glob.glob(os.path.join(input_dir, "*.webm"))

for webm_file in webm_files:
    #define output file name
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(webm_file))[0] + '.mp4')

    #Capture video frames 
    video = cv2.VideoCapture(webm_file)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    #Define the codec and create VideoWriter Object. The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width,frame_height))

    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            # write the frame
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

        else:
            break

    # When everything is done, release the video capture and video write objects
    video.release()
    out.release()

    #Closes all the frames
    cv2.destroyAllWindows()
