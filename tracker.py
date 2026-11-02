import cv2
import numpy as np
import imageio

cap = cv2.VideoCapture("input_traffic.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Feature detection parameters
feature_params = dict(maxCorners=150, qualityLevel=0.03, minDistance=10, blockSize=7)

# Deterministic colors for trails
np.random.seed(42)
colors = np.random.randint(50, 255, (150, 3)).tolist()

_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params) #shi-tomasi corner detection

mask = np.zeros_like(prev_frame)
gif_frames = []
frame_count = 0

print("Generating demo.gif...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is None or len(p0) < 20 or frame_count % 40 == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame) #refreshing points every 40 frames or when tracking drops

    'Lucas-Kanade optical flow'
    p1, status, _ = cv2.calcOpticalFlowPyrLK( 
        prev_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2
    )

    if p1 is not None and status is not None:
        good_new = p1[status.ravel() == 1]
        good_old = p0[status.ravel() == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            col = colors[i % len(colors)]
            mask = cv2.line(mask, (a, b), (c, d), col, 2)
            frame = cv2.circle(frame, (a, b), 4, col, -1)

        p0 = good_new.reshape(-1, 1, 2)

    prev_gray = frame_gray.copy()

    # Capture every 2nd frame at 720px width for smooth GIF playback
    if frame_count % 2 == 0:
        composite = cv2.add(frame, mask)
        h, w = composite.shape[:2]
        resized = cv2.resize(composite, (720, int(h * 720 / w)), interpolation=cv2.INTER_AREA)
        gif_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

cap.release()

# Save final GIF
imageio.mimsave("demo.gif", gif_frames, fps=fps // 2, loop=0, subrectangles=True)
print("Complete! demo.gif is ready.")