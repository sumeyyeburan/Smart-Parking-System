import cv2
import json
import os
import pandas as pd

class ROIManager:
    def __init__(self):
        self.parking_spots = {}
        self.spot_counter = 0

    def load_from_cnr_csv(self, csv_path, img_width=1000, img_height=750,
                           orig_width=2592, orig_height=1944):
        """CNR-EXT camera CSV dosyasından ROI koordinatlarını yükle ve ölçekle"""
        df = pd.read_csv(csv_path)
        x_scale = img_width / orig_width
        y_scale = img_height / orig_height

        self.parking_spots = {}
        for _, row in df.iterrows():
            slot_id = int(row["SlotId"])
            x = int(row["X"] * x_scale)
            y = int(row["Y"] * y_scale)
            w = int(row["W"] * x_scale)
            h = int(row["H"] * y_scale)

            self.parking_spots[slot_id] = {
                "coords": [x, y, w, h],
                "occupied": False,
                "yolo_coords": None
            }

        self.spot_counter = max(self.parking_spots.keys()) if self.parking_spots else 0
        print(f"{len(self.parking_spots)} park yeri CSV'den yuklendi ve olceklendi.")
        return self.parking_spots

    def load_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.parking_spots = {int(k): v for k, v in data.items()}
        self.spot_counter = max(self.parking_spots.keys()) if self.parking_spots else 0
        print(f"{len(self.parking_spots)} park yeri JSON'dan yüklendi.")
        return self.parking_spots

    def save_to_json(self, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.parking_spots, f, indent=2)
        print(f"Park yerleri kaydedildi: {json_path}")

    def draw_manual_rois(self, frame):
        self.parking_spots = {}
        self.spot_counter  = 0
        clone        = frame.copy()
        drawing      = False
        start_x, start_y = -1, -1
        temp_frame   = frame.copy()

        print("\nPark yerlerini çizmek için mouse ile dikdörtgen çiz.")
        print("Sol tık basılı tut -> sürükle -> bırak = park yeri ekle")
        print("'s' = kaydet ve çık | 'z' = son eklenen yeri sil | ESC = iptal\n")

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_x, start_y, temp_frame, clone

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_x, start_y = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_frame = clone.copy()
                    cv2.rectangle(temp_frame, (start_x, start_y), (x, y), (0, 255, 255), 2)
                    cv2.imshow("ROI Ciz", temp_frame)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1 = min(start_x, x), min(start_y, y)
                x2, y2 = max(start_x, x), max(start_y, y)
                w, h = x2 - x1, y2 - y1
                if w > 10 and h > 10:
                    self.spot_counter += 1
                    self.parking_spots[self.spot_counter] = {
                        "coords": [x1, y1, w, h],
                        "occupied": False,
                        "yolo_coords": None
                    }
                    print(f"  Park yeri #{self.spot_counter} eklendi: ({x1},{y1}) {w}x{h}")
                    clone = temp_frame.copy()
                    self._draw_spots(clone)
                    cv2.imshow("ROI Ciz", clone)

        cv2.namedWindow("ROI Ciz")
        cv2.setMouseCallback("ROI Ciz", mouse_callback)
        self._draw_spots(clone)
        cv2.imshow("ROI Ciz", clone)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                print(f"\nToplam {len(self.parking_spots)} park yeri kaydedildi.")
                break
            elif key == ord("z") and self.parking_spots:
                last_id = max(self.parking_spots.keys())
                del self.parking_spots[last_id]
                print(f"  Park yeri #{last_id} silindi.")
                clone = frame.copy()
                self._draw_spots(clone)
                cv2.imshow("ROI Ciz", clone)
            elif key == 27:
                self.parking_spots = {}
                break

        cv2.destroyAllWindows()
        return self.parking_spots

    def get_stats(self):
        total    = len(self.parking_spots)
        occupied = sum(1 for s in self.parking_spots.values() if s["occupied"])
        empty    = total - occupied
        rate     = (occupied / total * 100) if total > 0 else 0
        return {
            "total":          total,
            "occupied":       occupied,
            "empty":          empty,
            "occupancy_rate": round(rate, 1)
        }

    def draw_spots(self, frame):
        self._draw_spots(frame)
        return frame

    def _draw_spots(self, frame):
        for spot_id, spot in self.parking_spots.items():
            # Sadece dolu olanlari ciz, bos olanlari atla
            if not spot.get("occupied", False):
                continue

            # Dolu ise: YOLO'nun buldugu gercek arac sinirini kullan (varsa)
            if spot.get("yolo_coords") is not None:
                x1, y1, x2, y2 = [int(v) for v in spot["yolo_coords"]]
            else:
                x, y, w, h = [int(v) for v in spot["coords"]]
                x1, y1, x2, y2 = x, y, x + w, y + h

            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{spot_id}", (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)