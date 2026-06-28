import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
import os
import time
import json
import sys


# Çizim aracı ve arka plan görüntüsü için PIL eklendi
try:
    from streamlit_drawable_canvas import st_canvas
    from PIL import Image
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

sys.path.append(os.path.dirname(__file__))
from database import init_db, verify_user, get_all_users, add_user, delete_user, log_analysis, get_logs, get_stats_by_hour

# -- Ayarlar --
MODEL_PATH          = "models/parking_cnn.h5"
CSV_FOLDER          = "data"
IMG_SIZE            = (150, 150)
THRESHOLD           = 0.5
CONFIDENCE_OVERRIDE = 0.99
REFERENCE_FOLDER    = "data/references"
FULL_IMG_FOLDER     = "data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY"
VIDEO_ROI_JSON      = "data/annotations/video_rois.json"
UPLOAD_DIR          = "data/uploads" # Kullanıcıların yüklediği dataların ineceği klasör
VALID_CAMERAS       = ["camera3", "camera5", "camera6", "camera7", "camera8", "camera9"]

st.set_page_config(page_title="Otopark Doluluk Sistemi", layout="wide")

# Veritabanını ve Klasörleri başlat
init_db()
os.makedirs(os.path.dirname(VIDEO_ROI_JSON), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Model yükleme ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8s.pt")
    cnn_model  = tf.keras.models.load_model(MODEL_PATH)
    return yolo_model, cnn_model


# ── Görüntü işleme fonksiyonları ──────────────────────────────────────────────
def process_image(img, yolo_model, cnn_model, force_camera=None,
                  custom_spots=None, show_non_parked=True, tracker_history=None):
    img_h, img_w = img.shape[:2]

    results    = yolo_model(img, classes=[2, 3, 5, 7], verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append([x1, y1, x2, y2])

    spots = {}
    if custom_spots is not None:
        spots = custom_spots
    elif force_camera is not None:
        csv_path = os.path.join(CSV_FOLDER, f"{force_camera}.csv")
        if os.path.exists(csv_path):
            df      = pd.read_csv(csv_path)
            x_scale = img_w / 2592
            y_scale = img_h / 1944
            for _, row in df.iterrows():
                slot_id = int(row["SlotId"])
                x = int(row["X"] * x_scale)
                y = int(row["Y"] * y_scale)
                w = int(row["W"] * x_scale)
                h = int(row["H"] * y_scale)
                spots[slot_id] = {"coords": [x, y, w, h], "occupied": False, "yolo_coords": None}

    parked_yolo_boxes = []

    for spot_id, spot in spots.items():
        sx, sy, sw, sh     = spot["coords"]
        sx1, sy1, sx2, sy2 = sx, sy, sx + sw, sy + sh

        crop = img[max(0, sy1):min(img_h, sy2), max(0, sx1):min(img_w, sx2)]
        if crop.size == 0:
            continue

        test_img      = cv2.resize(crop, IMG_SIZE) / 255.0
        test_img      = np.expand_dims(test_img, axis=0)
        cnn_pred      = cnn_model.predict(test_img, verbose=0)[0][0]
        cnn_says_full = cnn_pred > THRESHOLD

        best_iou = 0
        best_box = None
        for dx1, dy1, dx2, dy2 in detections:
            ix1 = max(sx1, dx1)
            iy1 = max(sy1, dy1)
            ix2 = min(sx2, dx2)
            iy2 = min(sy2, dy2)
            inter_area  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            spot_area   = sw * sh
            if spot_area > 0:
                overlap_ratio = inter_area / spot_area
                if overlap_ratio > 0.3 and overlap_ratio > best_iou:
                    best_iou      = overlap_ratio
                    best_box      = [dx1, dy1, dx2, dy2]

        yolo_says_full       = best_box is not None
        is_bottom_blind_spot = sy > (img_h * 0.70)

        if cnn_says_full:
            if yolo_says_full:
                spot["occupied"]    = True
                spot["yolo_coords"] = best_box
                parked_yolo_boxes.append(tuple(best_box))
            elif is_bottom_blind_spot:
                spot["occupied"] = True
            elif cnn_pred > CONFIDENCE_OVERRIDE:
                spot["occupied"] = True
            else:
                spot["occupied"] = False

    annotated = img.copy()

    # Dolu (park edilmiş) araçları kırmızı çiz
    for spot_id, spot in spots.items():
        if not spot["occupied"]:
            continue
        if spot["yolo_coords"] is not None:
            x1, y1, x2, y2 = spot["yolo_coords"]
        else:
            x, y, w, h     = spot["coords"]
            x1, y1, x2, y2 = x, y, x + w, y + h
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"#{spot_id}", (x1+2, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Park dışı araçlar
    if show_non_parked:
        for dx1, dy1, dx2, dy2 in detections:
            if tuple([dx1, dy1, dx2, dy2]) not in parked_yolo_boxes:
                # Tracker history varsa hareket kontrolü yap
                is_moving = True
                if tracker_history is not None:
                    cx = (dx1 + dx2) // 2
                    cy = (dy1 + dy2) // 2
                    # Önceki frame ile merkez karşılaştır
                    for prev_cx, prev_cy in tracker_history:
                        dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2) ** 0.5
                        if dist < 15:  # 15 piksel eşiği — durmuş sayılır
                            is_moving = False
                            break

                if is_moving:
                    # Hareket halinde — sarı
                    cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
                    cv2.putText(annotated, "Hareket", (dx1, dy1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # Duruyorsa (park dışı ama hareketsiz) — hiçbir şey çizme

    total    = len(spots)
    occupied = sum(1 for s in spots.values() if s["occupied"])
    empty    = total - occupied
    rate     = (occupied / total * 100) if total > 0 else 0
    stats    = {"total": total, "occupied": occupied, "empty": empty, "occupancy_rate": round(rate, 1)}

    return annotated, stats


# ── Login & Register ekranı ───────────────────────────────────────────────────
def show_login():
    st.title("🚗 Otopark Doluluk Tespit Sistemi")
    
    tab_login, tab_register = st.tabs(["🔐 Giriş Yap", "📝 Kayıt Ol"])

    with tab_login:
        st.subheader("Sisteme Giriş")
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit   = st.form_submit_button("Giriş Yap")

        if submit:
            user = verify_user(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Kullanıcı adı veya şifre hatalı.")

    with tab_register:
        st.subheader("Yeni Kullanıcı Kaydı")
        with st.form("register_form"):
            new_username = st.text_input("Kullanıcı Adı Belirleyin")
            new_password = st.text_input("Şifrenizi Belirleyin", type="password")
            new_password_confirm = st.text_input("Şifrenizi Tekrar Girin", type="password")
            register_submit = st.form_submit_button("Hesap Oluştur")

        if register_submit:
            if not new_username or not new_password:
                st.error("Kullanıcı adı ve şifre boş bırakılamaz!")
            elif new_password != new_password_confirm:
                st.error("Girdiğiniz şifreler eşleşmiyor!")
            else:
                success, msg = add_user(new_username, new_password, "user")
                if success:
                    st.success("Kayıt işlemi başarılı! Lütfen 'Giriş Yap' sekmesine geçerek giriş yapınız.")
                else:
                    st.error(msg)


# ── Admin paneli ──────────────────────────────────────────────────────────────
def show_admin_panel(yolo_model, cnn_model):
    st.title("🔧 Admin Paneli")

    tab1, tab2, tab3 = st.tabs(["👥 Kullanıcılar", "📊 Tüm Raporlar / Loglar", "✏️ Veriseti ve Slot Çizimi"])

    with tab1:
        st.subheader("Kullanıcı Listesi")
        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)

        st.divider()
        st.subheader("Kullanıcı Ekle / Sil")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.form("add_user_form"):
                st.write("Yeni Kullanıcı Ekle")
                new_username = st.text_input("Kullanıcı Adı")
                new_password = st.text_input("Şifre", type="password")
                new_role     = st.selectbox("Rol", ["user", "admin"])
                add_submit   = st.form_submit_button("Ekle")

            if add_submit:
                success, msg = add_user(new_username, new_password, new_role)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        
        with col2:
            user_ids = {f"{u['username']} (ID:{u['id']})": u["id"] for u in users if u["username"] != "admin"}
            if user_ids:
                st.write("Kullanıcı Sil")
                selected_user = st.selectbox("Silinecek kullanıcı", list(user_ids.keys()))
                if st.button("🗑️ Sil", type="primary"):
                    delete_user(user_ids[selected_user])
                    st.success("Kullanıcı silindi.")
                    st.rerun()

    with tab2:
        st.subheader("Kullanıcı Hareketleri ve Analiz Kayıtları")
        logs = get_logs()
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs, use_container_width=True)
            
            st.divider()
            st.subheader("📈 Saatlik Ortalama Doluluk")
            cam_filter = st.selectbox("Kamera filtresi", ["Tümü"] + VALID_CAMERAS)
            camera_arg = None if cam_filter == "Tümü" else cam_filter
            hourly     = get_stats_by_hour(camera_arg)

            if hourly:
                df_hourly = pd.DataFrame(hourly)
                df_hourly["hour"] = df_hourly["hour"].astype(str) + ":00"
                st.bar_chart(df_hourly.set_index("hour")["avg_rate"])
        else:
            st.info("Henüz sistemde kaydedilmiş bir işlem yok.")

    with tab3:
        st.subheader("Kullanıcıların Yüklediği Dataset Dosyaları")
        
        uploaded_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
        
        if not uploaded_files:
            st.info("Sisteme henüz kullanıcı tarafından dosya yüklenmemiş.")
        else:
            if not HAS_CANVAS:
                st.warning("Bu özelliği kullanmak için terminalde `pip install streamlit-drawable-canvas` komutunu çalıştırıp uygulamayı yeniden başlatın.")
            else:
                bekleyenler = []
                tamamlananlar = []
                
                for f in uploaded_files:
                    roi_path = os.path.join("data/annotations", f"{f}_rois.json")
                    file_path = os.path.join(UPLOAD_DIR, f)
                    if os.path.exists(roi_path):
                        tamamlananlar.append((f, os.path.getmtime(file_path)))
                    else:
                        bekleyenler.append((f, os.path.getmtime(file_path)))

                bekleyenler.sort(key=lambda x: x[1], reverse=True)
                tamamlananlar.sort(key=lambda x: x[1], reverse=True)

                bekleyen_isimler = [x[0] for x in bekleyenler]
                tamamlanan_isimler = [x[0] for x in tamamlananlar]

                kategori = st.radio("Dosya Durumu:", ["Çizim Bekleyenler (Yeni)", "Tamamlananlar"], horizontal=True)
                
                secilebilir_dosyalar = bekleyen_isimler if kategori == "Çizim Bekleyenler (Yeni)" else tamamlanan_isimler
                
                if not secilebilir_dosyalar:
                    st.info("Bu kategoride dosya bulunmuyor.")
                else:
                    selected_file = st.selectbox("İşlem yapılacak dosyayı seçin (En yeniler üstte)", secilebilir_dosyalar)
                    file_path = os.path.join(UPLOAD_DIR, selected_file)
                    roi_json_path = os.path.join("data/annotations", f"{selected_file}_rois.json")

                    # --- DÜZELTME: bg_img tanımlamasını if bloklarının üzerine taşıdık ---
                    bg_img = None
                    is_video = selected_file.split('.')[-1].lower() in ['mp4', 'avi']
                    
                    if is_video:
                        cap = cv2.VideoCapture(file_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            bg_img = frame
                        else:
                            st.error("Videodan kare okunamadı!")
                    else:
                        bg_img = cv2.imread(file_path)
                    # ------------------------------------------------------------------

                    if kategori == "Tamamlananlar":
                        st.success("Bu dosya için daha önce park alanları çizilmiş.")
                        
                        # Mevcut ROI'leri görselleştir
                        if os.path.exists(roi_json_path):
                            with open(roi_json_path, "r") as f:
                                existing_rois = json.load(f)
                            
                            if bg_img is not None:
                                preview = bg_img.copy()
                                for spot_id, spot in existing_rois.items():
                                    x, y, w, h = spot["coords"]
                                    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(preview, f"#{spot_id}", (x+2, y+14),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                                         caption=f"{len(existing_rois)} slot çizilmiş",
                                         use_column_width=True)
                    else:
                        st.warning("Bu dosya yeni eklendi, park slotları henüz çizilmemiş.")

                    bg_img = None
                    is_video = selected_file.split('.')[-1].lower() in ['mp4', 'avi']
                    
                    if is_video:
                        cap = cv2.VideoCapture(file_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            bg_img = frame
                        else:
                            st.error("Videodan kare okunamadı!")
                    else:
                        bg_img = cv2.imread(file_path)

                    if bg_img is not None:
                        img_h, img_w = bg_img.shape[:2]
                        max_canvas_width = 800
                        
                        if img_w > max_canvas_width:
                            scale = max_canvas_width / img_w
                        else:
                            scale = 1.0
                            
                        canvas_w = int(img_w * scale)
                        canvas_h = int(img_h * scale)
                        
                        bg_img_resized = cv2.resize(bg_img, (canvas_w, canvas_h))
                        bg_img_rgb = cv2.cvtColor(bg_img_resized, cv2.COLOR_BGR2RGB)
                        bg_img_pil = Image.fromarray(bg_img_rgb)
                        
                        st.write(f"**{selected_file}** üzerine park alanlarını çizin:")
                        st.caption("Orijinal görüntü büyük olduğu için ekrana sığacak şekilde küçültülmüştür. Çizimleriniz orijinal boyuta göre kaydedilecektir.")
                        
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 0, 0, 0.3)",
                            stroke_width=2,
                            stroke_color="#ff0000",
                            background_image=bg_img_pil,
                            update_streamlit=True,
                            height=canvas_h,
                            width=canvas_w,
                            drawing_mode="rect",
                            key="canvas_" + selected_file,
                        )

                        if st.button("Çizimleri Dataset İçin Kaydet"):
                            if canvas_result.json_data is not None:
                                custom_spots = {}
                                objects = canvas_result.json_data["objects"]
                                for i, obj in enumerate(objects):
                                    x = int((obj["left"]) / scale)
                                    y = int((obj["top"]) / scale)
                                    w = int((obj["width"] * obj["scaleX"]) / scale)
                                    h = int((obj["height"] * obj["scaleY"]) / scale)
                                    
                                    custom_spots[str(i+1)] = {"coords": [x, y, w, h], "occupied": False, "yolo_coords": None}
                                
                                with open(roi_json_path, "w") as f:
                                    json.dump(custom_spots, f)
                                st.success(f"{len(objects)} adet park alanı başarıyla kaydedildi! Kullanıcılar artık bu dosyadan anında çıktı alabilir.")
                                st.rerun()


# ── Kullanıcı paneli ──────────────────────────────────────────────────────────
def show_user_panel(yolo_model, cnn_model):
    user = st.session_state["user"]
    st.title("🚗 Otopark Doluluk Sistemi")
    st.caption(f"Hoş geldiniz, **{user['username']}**")

    tab1, tab2, tab3, tab4 = st.tabs(["📷 Dataset Görüntüsü", "🎬 Video Simülasyonu", "📊 Raporlarım", "📂 Harici Dosya Yükle"])

    # ── YENİ: Otomatik Dataset Dosyası Listeleme ──
    with tab1:
        st.subheader("Datasetten Görüntü Seç ve Analiz Et")
        
        if not os.path.exists(FULL_IMG_FOLDER):
            st.error(f"Klasör bulunamadı: {FULL_IMG_FOLDER}")
        else:
            dates = sorted(os.listdir(FULL_IMG_FOLDER))
            selected_date = st.selectbox("Tarih seç", dates, key="img_date")

            if selected_date:
                cam_folder   = os.path.join(FULL_IMG_FOLDER, selected_date)
                cameras      = [c for c in sorted(os.listdir(cam_folder)) if c in VALID_CAMERAS]
                selected_cam = st.selectbox("Kamera seç", cameras, key="img_cam")

                if selected_cam:
                    img_folder = os.path.join(cam_folder, selected_cam)
                    images     = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
                    
                    if not images:
                        st.info("Bu kameraya ait görüntü bulunamadı.")
                    else:
                        selected_img = st.selectbox("Görüntü seç (Dataset içinden)", images, key="img_file")
                        
                        if st.button("Analiz Et"):
                            img_path = os.path.join(img_folder, selected_img)
                            img      = cv2.imread(img_path)

                            with st.spinner("Analiz yapılıyor..."):
                                annotated, stats = process_image(img, yolo_model, cnn_model, force_camera=selected_cam, show_non_parked=False)
                                log_analysis(user["id"], user["username"], selected_cam, "image", stats)

                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Kamera: {selected_cam} - Görüntü: {selected_img}", use_column_width=True)
                            with col2:
                                st.subheader("📊 Sonuçlar")
                                st.metric("Kamera",          selected_cam)
                                st.metric("Toplam Park Yeri", stats["total"])
                                st.metric("Dolu",             stats["occupied"])
                                st.metric("Boş",              stats["empty"])
                                st.metric("Doluluk Oranı",    f"%{stats['occupancy_rate']}")

    with tab2:
        show_video_simulation(yolo_model, cnn_model, is_admin=False)

    with tab3:
        st.subheader("Analiz Geçmişim")
        logs = get_logs(user_id=user["id"])
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs[["timestamp", "camera", "source_type", "total_spots", "occupied", "empty", "occupancy_rate"]], use_container_width=True)

    with tab4:
        st.subheader("Kendi Verinizi Ekleyin veya Seçin")
        st.write("Bilgisayarınızdan yeni bir dosya yükleyebilir veya daha önce yüklenip admin tarafından eğitilmiş dataset dosyanızı seçebilirsiniz.")
        
        islem_tipi = st.radio("İşlem Tipi:", ["Bilgisayardan Yeni Yükle", "Sistemdeki (Eğitilmiş) Dosyalardan Seç"], horizontal=True)

        selected_filename = None
        file_path_to_process = None

        if islem_tipi == "Bilgisayardan Yeni Yükle":
            ext_file = st.file_uploader("Fotoğraf veya video yükleyin", type=["jpg", "jpeg", "png", "mp4", "avi"])
            if ext_file is not None:
                selected_filename = ext_file.name
                file_path_to_process = os.path.join(UPLOAD_DIR, selected_filename)
                with open(file_path_to_process, "wb") as f:
                    f.write(ext_file.getbuffer())
                st.success(f"Dosya '{selected_filename}' sisteme eklendi.")

        elif islem_tipi == "Sistemdeki (Eğitilmiş) Dosyalardan Seç":
            mevcut_dosyalar = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
            if mevcut_dosyalar:
                selected_filename = st.selectbox("Dataset İçerisindeki Dosyalar", mevcut_dosyalar)
                file_path_to_process = os.path.join(UPLOAD_DIR, selected_filename)
            else:
                st.info("Sistemde henüz kayıtlı harici bir dosya yok.")

        if selected_filename and file_path_to_process:
            roi_json_path = os.path.join("data/annotations", f"{selected_filename}_rois.json")
            
            if os.path.exists(roi_json_path):
                st.success("✅ Admin tarafından bu dosya için park alanları tanımlanmış! Doğrudan analiz ediliyor.")
                
                with open(roi_json_path, "r") as f:
                    custom_rois = json.load(f)

                is_video = selected_filename.split('.')[-1].lower() in ['mp4', 'avi']

                if not is_video:
                    img = cv2.imread(file_path_to_process)
                    with st.spinner("Analiz ediliyor..."):
                        annotated, stats = process_image(img, yolo_model, cnn_model, custom_spots=custom_rois)
                    
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analiz Çıktısı", use_column_width=True)
                    st.success(f"Analiz Tamamlandı: Toplam: {stats['total']}, Dolu: {stats['occupied']}")
                    log_analysis(user["id"], user["username"], "Harici Dataset", "image", stats)
                
                else:
                    if st.button("▶️ Videoyu Analiz Et"):
                        cap = cv2.VideoCapture(file_path_to_process)
                        stframe = st.empty()
                        stats_placeholder = st.empty()
                        
                        frame_skip = 3
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            if frame_count % frame_skip != 0:
                                continue
                            
                            annotated, stats = process_image(frame, yolo_model, cnn_model, custom_spots=custom_rois)
                            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                            
                            stats_placeholder.markdown(f"""
                            | Toplam Alan | Dolu | Boş | Doluluk Oranı |
                            |:---:|:---:|:---:|:---:|
                            | **{stats['total']}** | **{stats['occupied']}** | **{stats['empty']}** | **%{stats['occupancy_rate']}** |
                            """)
                        
                        cap.release()
                        st.success("Video analizi tamamlandı.")
                        log_analysis(user["id"], user["username"], "Harici Dataset", "video", stats)

            else:
                st.warning("⏳ Bu dosya için Admin tarafından henüz park alanı (ROI) çizilmemiş. Yüklemeniz Admin paneline (Çizim Bekleyenler) iletildi.")
                
                if st.button("Yine de Sadece Araç Tespiti (Alan Dışı) Yap"):
                    is_video = selected_filename.split('.')[-1].lower() in ['mp4', 'avi']
                    if not is_video:
                        img = cv2.imread(file_path_to_process)
                        annotated, stats = process_image(img, yolo_model, cnn_model,
                                  custom_spots=custom_rois,
                                  show_non_parked=True,
                                  tracker_history=None)
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        cap = cv2.VideoCapture(file_path_to_process)
                        stframe = st.empty()
                        
                        frame_skip = 3
                        frame_count = 0
                        
                        tracker_history = []
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_count += 1
                            if frame_count % frame_skip != 0:
                               continue

                            annotated, stats = process_image(frame, yolo_model, cnn_model,
                                      custom_spots=custom_rois,
                                      show_non_parked=True,
                                      tracker_history=tracker_history)

                            # Mevcut araç merkezlerini geçmişe ekle
                            results = yolo_model(frame, classes=[2, 3, 5, 7], verbose=False)
                            tracker_history = []
                            for r in results:
                                for box in r.boxes:
                                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                                    tracker_history.append(((x1+x2)//2, (y1+y2)//2))

                            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_column_width=True)
                        cap.release()


# ── Video simülasyonu ──────────────────────────────
def show_video_simulation(yolo_model, cnn_model, is_admin=False):
    st.subheader("🎬 Görüntü Serisi (Video Simülasyonu)")

    if not os.path.exists(FULL_IMG_FOLDER):
        st.error(f"Klasör bulunamadı: {FULL_IMG_FOLDER}")
        return

    dates         = sorted(os.listdir(FULL_IMG_FOLDER))
    selected_date = st.selectbox("Tarih seç", dates, key="sim_date")

    if selected_date:
        cam_folder   = os.path.join(FULL_IMG_FOLDER, selected_date)
        cameras      = [c for c in sorted(os.listdir(cam_folder)) if c in VALID_CAMERAS]
        selected_cam = st.selectbox("Kamera seç", cameras, key="sim_cam")

        if selected_cam:
            img_folder = os.path.join(cam_folder, selected_cam)
            images     = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
            st.write(f"{len(images)} görüntü bulundu.")
            delay = st.slider("Görüntüler arası gecikme (saniye)", 0.1, 0.5, 0.1,0.1, key="sim_delay")

            if st.button("▶️ Oynat", key="sim_play"):
                user = st.session_state.get("user", {})
                img_placeholder   = st.empty()
                stats_placeholder = st.empty()
                progress_bar      = st.progress(0)

                for i, fname in enumerate(images):
                    img_path = os.path.join(img_folder, fname)
                    img      = cv2.imread(img_path)
                    if img is None:
                        continue

                    annotated, stats = process_image(
                        img, yolo_model, cnn_model, force_camera=selected_cam,show_non_parked=False 
                    )

                    if user:
                        log_analysis(user["id"], user["username"], selected_cam, "simulation", stats)

                    img_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"[{i+1}/{len(images)}] {fname} — Kamera: {selected_cam}",
                        use_column_width=True
                    )
                    stats_placeholder.markdown(f"""
| | |
|---|---|
| **Kamera** | {selected_cam} |
| **Toplam** | {stats['total']} |
| **Dolu** | {stats['occupied']} |
| **Boş** | {stats['empty']} |
| **Doluluk** | %{stats['occupancy_rate']} |
                    """)
                    progress_bar.progress((i + 1) / len(images))
                    time.sleep(delay)

                st.success("Seri tamamlandı!")


# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"]:
        user = st.session_state["user"]
        st.sidebar.title("👤 Kullanıcı")
        st.sidebar.write(f"**{user['username']}**")
        st.sidebar.write(f"Rol: `{user['role']}`")
        if st.sidebar.button("🚪 Çıkış Yap"):
            st.session_state["user"] = None
            st.rerun()

    if not st.session_state["user"]:
        show_login()
    else:
        yolo_model, cnn_model = load_models()
        user = st.session_state["user"]

        if user["role"] == "admin":
            show_admin_panel(yolo_model, cnn_model)
        else:
            show_user_panel(yolo_model, cnn_model)


if __name__ == "__main__":
    main()