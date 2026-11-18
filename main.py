"""
ADAS Event Simulator - Korak 3 (v2): FCW i Pedestrian Warning
LDW je uklonjen iz alarma/logovanja (monokularna detekcija trake se pokazala
nepouzdanom za pouzdano okidanje upozorenja) - traka i dalje ostaje vizuelizovana,
samo se ne koristi za alarm. FCW i Pedestrian Warning rade preko YOLOv8 detekcija.
"""

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import sys
import csv

# Klase iz COCO dataset-a koje nas zanimaju (YOLOv8 je vec treniran na ovima)
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}

# Boje za bounding box-ove (BGR format, jer OpenCV tako radi)
COLORS = {
    "person": (0, 0, 255),       # crvena - pesaci
    "bicycle": (255, 255, 0),
    "car": (0, 255, 0),          # zelena - vozila
    "motorcycle": (255, 255, 0),
    "bus": (0, 255, 0),
    "truck": (0, 255, 0),
}

YOLOP_INPUT_SIZE = 320  # mora da se poklapa sa imenom .onnx fajla (yolop-320-320.onnx)

# --- Pragovi za ADAS dogadjaje (slobodno podesavaj ako precesto/preretko okidaju) ---
FCW_MIN_HEIGHT_RATIO = 0.25         # vozilo zauzima >25% visine frejma -> blizu je
PEDESTRIAN_MIN_HEIGHT_RATIO = 0.15  # pesak zauzima >15% visine frejma -> blizu je
DANGER_ZONE_LEFT = 0.30             # centralna zona ispred nas (30%-70% sirine frejma)
DANGER_ZONE_RIGHT = 0.70


def detect_lane_yolop(frame, session, input_name):
    """Koristi YOLOP model (pretrained na BDD100K) za segmentaciju linije trake.
    Vraca binarnu masku (0/255) iste velicine kao originalni frejm.
    Koristi se samo za vizuelizaciju, ne za alarm (vidi napomenu na vrhu fajla)."""
    orig_h, orig_w = frame.shape[:2]

    resized = cv2.resize(frame, (YOLOP_INPUT_SIZE, YOLOP_INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std

    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    outputs = session.run(None, {input_name: input_tensor})
    ll_seg_out = outputs[2]  # [1, 2, 320, 320] - kanal 0: pozadina, kanal 1: linija

    lane_mask = (ll_seg_out[0, 1] > ll_seg_out[0, 0]).astype(np.uint8) * 255
    lane_mask = cv2.resize(lane_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return lane_mask


def draw_lane_mask(frame, lane_mask):
    """Oboji detektovane piksele linije trake preko frejma (poluprovidno, zuta boja)."""
    overlay = frame.copy()
    overlay[lane_mask > 0] = (0, 255, 255)  # zuta u BGR
    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)


def check_fcw(x1, y1, x2, y2, label, frame_width, frame_height):
    """Da li ovo vozilo opravdava Forward Collision Warning?"""
    if label not in VEHICLE_LABELS:
        return False
    box_center_x = (x1 + x2) / 2
    box_height = y2 - y1

    in_danger_zone = (DANGER_ZONE_LEFT * frame_width) < box_center_x < (DANGER_ZONE_RIGHT * frame_width)
    is_close = (box_height / frame_height) > FCW_MIN_HEIGHT_RATIO
    return in_danger_zone and is_close


def check_pedestrian(x1, y1, x2, y2, label, frame_width, frame_height):
    """Da li ovaj pesak opravdava Pedestrian Warning?"""
    if label != "person":
        return False
    box_center_x = (x1 + x2) / 2
    box_height = y2 - y1

    in_danger_zone = (DANGER_ZONE_LEFT * frame_width) < box_center_x < (DANGER_ZONE_RIGHT * frame_width)
    is_close = (box_height / frame_height) > PEDESTRIAN_MIN_HEIGHT_RATIO
    return in_danger_zone and is_close


def draw_alerts(frame, alert_lines):
    """Iscrtava listu tekstova upozorenja u gornjem levom uglu, jedan ispod drugog."""
    for i, text in enumerate(alert_lines):
        y = 60 + i * 35
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main(video_path):
    yolo_model = YOLO("yolov8n.pt")

    yolop_session = ort.InferenceSession("yolop-320-320.onnx")
    yolop_input_name = yolop_session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Greska: ne mogu da otvorim video '{video_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    log_file = open("events_log.csv", "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "vreme_sekunde", "dogadjaj"])

    # Stanje za brojanje "incidenata" (ne svaki frejm dok upozorenje traje, samo pocetak)
    fcw_active = False
    ped_active = False
    fcw_count = 0
    ped_count = 0

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        timestamp = frame_count / fps

        # --- Detekcije na cistom frejmu, pre bilo kakvog crtanja ---
        lane_mask = detect_lane_yolop(frame, yolop_session, yolop_input_name)
        results = yolo_model(frame, verbose=False)[0]

        # --- Provera ADAS dogadjaja ---
        fcw_triggered = False
        ped_triggered = False

        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in RELEVANT_CLASSES:
                continue
            label = RELEVANT_CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if check_fcw(x1, y1, x2, y2, label, frame_width, frame_height):
                fcw_triggered = True
            if check_pedestrian(x1, y1, x2, y2, label, frame_width, frame_height):
                ped_triggered = True

        # --- Belezi samo pocetak incidenta (rastuca ivica), ne svaki frejm ---
        if fcw_triggered and not fcw_active:
            fcw_count += 1
            log_writer.writerow([frame_count, f"{timestamp:.2f}", "FCW"])
        if ped_triggered and not ped_active:
            ped_count += 1
            log_writer.writerow([frame_count, f"{timestamp:.2f}", "PEDESTRIAN"])

        fcw_active = fcw_triggered
        ped_active = ped_triggered

        # --- Sad crtamo sve preko frejma ---
        frame = draw_lane_mask(frame, lane_mask)  # samo vizuelno, ne koristi se za alarm

        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in RELEVANT_CLASSES:
                continue
            label = RELEVANT_CLASSES[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = COLORS[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {confidence:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        alert_lines = []
        if fcw_triggered:
            alert_lines.append("! FORWARD COLLISION WARNING")
        if ped_triggered:
            alert_lines.append("! PEDESTRIAN WARNING")
        draw_alerts(frame, alert_lines)

        cv2.putText(
            frame, f"Frame: {frame_count}", (10, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow("ADAS Simulator - Korak 3", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

    print("\n--- Sazetak ---")
    print(f"Forward Collision Warning: {fcw_count} put(a)")
    print(f"Pedestrian Warning: {ped_count} put(a)")
    print("Detalji upisani u events_log.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Upotreba: python main.py putanja_do_videa.mp4")
        sys.exit(1)

    main(sys.argv[1])