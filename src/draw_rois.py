import cv2
import json
import os

IMAGE_PATH = "data/references/video_frame.jpg"
OUTPUT_JSON = "data/annotations/video_rois.json"

parking_spots = {}
spot_counter  = 0
clone         = None
drawing       = False
start_x, start_y = -1, -1
temp_frame    = None

def draw_spots(frame):
    for spot_id, spot in parking_spots.items():
        x, y, w, h = spot["coords"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"#{spot_id}", (x+2, y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, temp_frame, clone, spot_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = clone.copy()
            draw_spots(temp_frame)
            cv2.rectangle(temp_frame, (start_x, start_y), (x, y), (0, 255, 255), 2)
            cv2.imshow("ROI Ciz", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)
        w, h = x2 - x1, y2 - y1
        if w > 10 and h > 10:
            spot_counter += 1
            parking_spots[spot_counter] = {"coords": [x1, y1, w, h], "occupied": False}
            print(f"  Park yeri #{spot_counter} eklendi: ({x1},{y1}) {w}x{h}")
            clone = temp_frame.copy()
            draw_spots(clone)
            cv2.imshow("ROI Ciz", clone)

def main():
    global clone, temp_frame

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"HATA: {IMAGE_PATH} bulunamadi!")
        return

    # Dikey video ise yatay çevir (gösterim için)
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print("Görüntü yatay çevrildi.")

    clone      = img.copy()
    temp_frame = img.copy()

    print("\nTalimatlar:")
    print("  Sol tık basılı tut + sürükle = park yeri ekle")
    print("  's' = kaydet ve çık")
    print("  'z' = son eklenen yeri sil")
    print("  ESC = iptal\n")

    cv2.namedWindow("ROI Ciz", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Ciz", 1200, 700)
    cv2.setMouseCallback("ROI Ciz", mouse_callback)
    draw_spots(clone)
    cv2.imshow("ROI Ciz", clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(parking_spots, f, indent=2)
            print(f"\n{len(parking_spots)} park yeri kaydedildi: {OUTPUT_JSON}")
            break
        elif key == ord("z") and parking_spots:
            last_id = max(parking_spots.keys())
            del parking_spots[last_id]
            print(f"  Park yeri #{last_id} silindi.")
            clone = img.copy()
            draw_spots(clone)
            cv2.imshow("ROI Ciz", clone)
        elif key == 27:
            print("İptal edildi.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()