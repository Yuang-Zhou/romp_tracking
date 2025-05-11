import romp
import cv2
import torch
from ultralytics import YOLO
import numpy as np 

SMPL_24_EDGES = [
    # (0,1), (1,4), (4,7), (7,10),  # Left leg
    # (0,2), (2,5), (5,8), (8,11),  # Right leg
    # (0,3), (3,6), (6,12), (12,53),  # Spine to head
    # (6,16), (16,18), (18,20), (20,22),  # Left arm
    (6,17), (17,19), (19,21), (21,23)   # Right arm
]



if __name__ == '__main__':

    #settings = romp.main.default_settings
    settings = romp.main.romp_settings([
        '--smpl_path', '/Users/yuangzhou/PycharmProjects/romp_tracking-main/romp_model/SMPLA_NEUTRAL.pth',
        '--model_path', '/Users/yuangzhou/PycharmProjects/romp_tracking-main/romp_model/ROMP.pkl'
    ])

    cap = cv2.VideoCapture(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    romp_model = romp.ROMP(settings).to(device)

    detector = YOLO('yolo11n.pt')



    while True:
        ret, frame = cap.read()
        if not isinstance(frame, np.ndarray):
            print(" Frame is not a NumPy array. It's:", type(frame))
            # Try to convert or raise an error
            # raise
        
        if not ret:
            break

        # Run inference
        results = detector(frame)[0]
        # print(f'{results = }')

        # frame = torch.tensor(frame).to(device)

        # Draw bounding boxes for 'person' only
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if detector.names[cls] == "person":
                bbox = box.xyxy
                x1, y1, x2, y2 = map(int, bbox[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        hmr_output = romp_model(frame)

        if hmr_output is not None:

            canvas_size = 512
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

            # Get 2D joint coordinates (x, y) of SMPL-24
            joints = hmr_output['joints'][0][:, :2]  # shape (71, 2)
            joints = joints[0:54]  # Only up to index 53 (SMPL_24 defined)

            # Normalize joints to [0, 1] based on min/max
            min_xy = joints.min(axis=0)
            max_xy = joints.max(axis=0)
            range_xy = np.maximum(max_xy - min_xy, 1)  # avoid division by 0

            joints_norm = (joints - min_xy) / range_xy  # scale to [0,1]
            joints_canvas = (joints_norm * (canvas_size * 0.8)).astype(np.int32)  # scale to canvas
            joints_canvas += (canvas_size // 10, canvas_size // 10)  # center in canvas

            # Draw bones
            for i, j in SMPL_24_EDGES:
                pt1, pt2 = tuple(joints_canvas[i]), tuple(joints_canvas[j])
                cv2.line(canvas, pt1, pt2, color=(0, 255, 0), thickness=2)

            # Draw joints
            for i in set(sum(SMPL_24_EDGES, ())):
                x, y = joints_canvas[i]
                cv2.circle(canvas, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.imshow("SMPL 24 Skeleton", canvas)




        # Show both windows
        cv2.imshow("Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

