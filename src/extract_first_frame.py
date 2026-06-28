import cv2

cap = cv2.VideoCapture("data/videos/test_video.mp4")
ret, frame = cap.read()
if ret:
    cv2.imwrite("data/references/video_frame.jpg", frame)
    print(f"Frame kaydedildi: {frame.shape}")
cap.release()