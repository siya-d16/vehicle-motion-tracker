import cv2
import numpy as np

#FOR ORB MATCHING :

# 1. Load two consecutive video frames
cap = cv2.VideoCapture("input_traffic.mp4")
ret1, frame1 = cap.read()
for _ in range(5):
    ret2, frame2 = cap.read()
cap.release()

if not ret1 or not ret2:
    raise ValueError("Could not read frames from video.")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# 2. Keypoint Detection & Description using ORB
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 3. Match Descriptors using Brute-Force Matcher with Hamming Distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
raw_matches = bf.knnMatch(des1, des2, k=2)

# 4. Filter Ambiguous Matches via Lowe's Ratio Test
good_matches = []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 5. Outlier Rejection via Geometric Verification (RANSAC)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography matrix to isolate consistent planar motion
_, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = inliers_mask.ravel().tolist()

# 6. Visualize Verified Matches and Save
draw_params = dict(
    matchColor=(0, 255, 0),        # Inlier matches in green
    singlePointColor=(0, 0, 255),  # Keypoints in red
    matchesMask=matches_mask,      # Only draw verified inliers
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

matched_img = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, **draw_params)

# Resize for clean viewing
h, w = matched_img.shape[:2]
matched_img = cv2.resize(matched_img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)

cv2.imwrite("orb_matches.jpg", matched_img)
print(f"Total Raw Matches: {len(raw_matches)}")
print(f"Lowe's Ratio Inliers: {len(good_matches)}")
print(f"RANSAC Verified Inliers: {sum(matches_mask)}")
print("Saved visualization to orb_matches.jpg!")