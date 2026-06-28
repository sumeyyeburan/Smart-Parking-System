import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
from roi_manager import ROIManager

# -- Ayarlar --
MODEL_PATH = "models/parking_cnn.h5"
IMAGE_PATH = "data/test2.jpg"
CSV_FOLDER = "data"
IMG_SIZE   = (150, 150)
THRESHOLD  = 0.4


def identify_camera(img, yolo_detections, csv_folder="data"):
    """
    9 kameranin CSV'sini sirayla dener, YOLO tespitleriyle en cok
    ortusen CSV'yi (dogru kamerayi) secer.
    """
    img_h, img_w = img.shape[:2]
    best_cam = None
    best_score = -1

    for cam_id in range(1, 10):
        csv_path = os.path.join(csv_folder, f"camera{cam_id}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        x_scale = img_w / 2592
        y_scale = img_h / 1944

        match_count = 0
        for _, row in df.iterrows():
            sx = int(row["X"] * x_scale)
            sy = int(row["Y"] * y_scale)
            sw = int(row["W"] * x_scale)
            sh = int(row["H"] * y_scale)
            sx2, sy2 = sx + sw, sy + sh
            spot_area = sw * sh
            if spot_area == 0:
                continue

            for dx1, dy1, dx2, dy2 in yolo_detections:
                ix1, iy1 = max(sx, dx1), max(sy, dy1)
                ix2, iy2 = min(sx2, dx2), min(sy2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / spot_area > 0.3:
                    match_count += 1
                    break

        score = match_count / len(df) if len(df) > 0 else 0
        print(f"  camera{cam_id}: {match_count}/{len(df)} eslesme (skor: {score:.2f})")

        if score > best_score:
            best_score = score
            best_cam = f"camera{cam_id}"

    return best_cam


def main():
    print("Modeller yukleniyor (YOLOv8s & Keras CNN)...")
    yolo_model = YOLO("yolov8s.pt")
    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"HATA: Goruntu bulunamadi -> {IMAGE_PATH}")
        return

    img_h, img_w = img.shape[:2]

    print("YOLO tespitleri yapiliyor...")
    results = yolo_model(img, classes=[2, 3, 5, 7], verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append([x1, y1, x2, y2])

    print(f"YOLO {len(detections)} arac buldu. Kamera tespiti yapiliyor...")
    cam_name = identify_camera(img, detections, CSV_FOLDER)
    if not cam_name:
        print("Kamera tespit edilemedi! Islem durduruluyor.")
        return

    dynamic_csv_path = os.path.join(CSV_FOLDER, f"{cam_name}.csv")
    print(f"Tespit edilen kamera: {cam_name}")
    print(f"Kullanilacak CSV dosyasi: {dynamic_csv_path}")

    roi = ROIManager()
    roi.load_from_cnr_csv(dynamic_csv_path)

    # Hibrit Karar Mekanizmasi (CNN + YOLO Bolgesel Filtresi)
    print("CNN dogrulamasi ve hibrit analiz uygulaniyor...")
    for spot_id, spot in roi.parking_spots.items():
        sx, sy, sw, sh = spot["coords"]
        sx1, sy1, sx2, sy2 = sx, sy, sx + sw, sy + sh

        spot["occupied"] = False
        spot["yolo_coords"] = None

        # 1. CNN tahmini
        crop = img[max(0, sy1):min(img_h, sy2), max(0, sx1):min(img_w, sx2)]
        if crop.size == 0:
            continue

        test_img = cv2.resize(crop, IMG_SIZE) / 255.0
        test_img = np.expand_dims(test_img, axis=0)

        cnn_pred = model.predict(test_img, verbose=0)[0][0]
        cnn_says_full = cnn_pred > THRESHOLD

        # 2. YOLO bu koordinatlarda arac buldu mu?
        best_iou = 0
        best_box = None
        for dx1, dy1, dx2, dy2 in detections:
            ix1, iy1 = max(sx1, dx1), max(sy1, dy1)
            ix2, iy2 = min(sx2, dx2), min(sy2, dy2)

            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            spot_area = sw * sh

            if spot_area > 0:
                overlap_ratio = inter_area / spot_area
                if overlap_ratio > 0.3 and overlap_ratio > best_iou:
                    best_iou = overlap_ratio
                    best_box = [dx1, dy1, dx2, dy2]

        yolo_says_full = best_box is not None

        # 3. Nihai karar agaci
        is_bottom_blind_spot = sy > (img_h * 0.70)

        if cnn_says_full:
            if yolo_says_full:
                spot["occupied"] = True
                spot["yolo_coords"] = best_box
            elif is_bottom_blind_spot:
                spot["occupied"] = True
            elif cnn_pred > 0.99:
                # CNN cok yuksek guvenle "dolu" diyor, YOLO gormese de kabul et
                # (dal/golge gibi orten nesneler YOLO'yu yanitabilir)
                spot["occupied"] = True
            else:
                spot["occupied"] = False    

    img = roi.draw_spots(img)
    stats = roi.get_stats()

    print("\n-- Sonuclar --")
    print(f"Toplam : {stats['total']}")
    print(f"Dolu   : {stats['occupied']}")
    print(f"Bos    : {stats['empty']}")
    print(f"Doluluk: %{stats['occupancy_rate']}")

    cv2.rectangle(img, (10, 10), (260, 130), (0, 0, 0), -1)
    cv2.putText(img, f"Toplam : {stats['total']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Dolu   : {stats['occupied']}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"Bos    : {stats['empty']}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Doluluk: %{stats['occupancy_rate']}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/sonuc5.jpg", img)
    print("\nSonuc kaydedildi: output/sonuc.jpg")

    cv2.imshow("Otopark Doluluk Sistemi", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()