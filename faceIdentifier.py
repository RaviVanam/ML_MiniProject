import cv2
import numpy as np
import face_recognition as fr

# This function assumes that there is only one face in the area defined by x1,y1,x2,y2
def findFace(targetEncoding, img, x1, y1, x2, y2):

    mask = np.zeros(img.shape[:2], np.uint8)
    # print(x1, x2, y1, y2)
    # mask[100:250, 150:450] = 255
    mask[y1:y2, x1:x2] = 255

    maskedImg = cv2.bitwise_and(img, img, mask=mask)
    maskedImg = cv2.resize(maskedImg, (0, 0), None, 0.25, 0.25)
    maskedImg = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2RGB)
    facesCurFrame = fr.face_locations(maskedImg)
    print(len(facesCurFrame))
    if len(facesCurFrame) == 0:
        return None

    encodesCurFrame = fr.face_encodings(maskedImg, facesCurFrame)
    curEncoding = encodesCurFrame[0]

    matches = fr.compare_faces([targetEncoding], curEncoding)
    faceDiss = fr.face_distance([targetEncoding], curEncoding)
    # matchIndex = np.argmin(faceDis)
    faceDis = faceDiss[0]

    if faceDis > 0.65:
        return "Not target"
    if matches[0] and faceDis < 0.45:
        return "target"

    return None