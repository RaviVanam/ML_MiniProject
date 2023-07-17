import cv2
import os

def createVideo(target_ids, id_frames):
    target_frames = []
    for id in target_ids:
        target_frames.extend(id_frames[id])

    imgs = [cv2.imread(os.path.join("outputImgs", image)) for image in
            sorted(os.listdir("outputImgs"), key=lambda x: int(x[:-4]))]

    fourcc = cv2.VideoWriter_fourcc('V', 'P', '8', '0')
    vw = cv2.VideoWriter("static/outputVid/output.webm", fourcc, 30, (1280, 720))

    for f in target_frames:
        vw.write(imgs[f])

    vw.release()