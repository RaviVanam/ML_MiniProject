import math
import cv2
import cvzone
import face_recognition as fr
from ultralytics import YOLO
from sort import *
from faceIdentifier import findFace
from videoCreator import createVideo

def get_target_encoding():
    # load face image
    img = cv2.imread("inputImg/input.jpg")
    # locate face in the image
    faces_cur_frame = fr.face_locations(img)
    # calculate the face encoding
    encodes_cur_frame = fr.face_encodings(img, faces_cur_frame)
    cur_encoding = encodes_cur_frame[0]
    return cur_encoding


def run():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("inputVid/input.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)

    # initializing yolo model
    model = YOLO("yolo-weights/yolov8n.pt")

    # initializing tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # calculating the target face encoding
    target_encoding = get_target_encoding()

    # list of ids corresponding to the target person
    target_ids = []

    # set of ids that have already been identified as target or not target
    id_identified = set()

    # dictionary of mappings from object ids to frame numbers
    id_frames = {}

    # current frame number
    f = -1

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        f = f + 1

        # saving the current frame with frame number as its name
        cv2.imwrite(os.path.join("outputImgs", "%d.jpg" % f), frame)

        # skipping 5 frames
        if f % 5 != 0:
            continue

        # running the current frame through YOLO model
        results = model(frame, stream=True)

        # initializing numpy array of empty detections
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:

                # only detecting people
                if box.cls[0] != 0:
                    continue

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))

                # confidence
                conf = math.ceil(box.conf[0] * 100) / 100

                # class name
                cls = int(box.cls[0])

                # updating dets
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

        tracker_results = tracker.update(detections)

        for result in tracker_results:
            # extracting coordinates and id of detected object from tracker result
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1

            # mapping id to frame number
            if id not in id_frames:
                id_frames[id] = []

            id_frames[id].append(f)

            # if the id is not mapped to someone then run face recognition on it and map the id to the name
            name = ""
            if id not in id_identified:
                name = findFace(target_encoding, frame, x1, y1, x2, y2)
                if name == "target":
                    target_ids.append(id)
                if name is not None:
                    id_identified.add(id)

            cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255, 0, 0))
            cvzone.putTextRect(frame, f'{id} {name}', (max(0, x1), max(25, y1)), scale=1, thickness=1)
            print(result, id)

        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    # create final video
    createVideo(target_ids, id_frames)
