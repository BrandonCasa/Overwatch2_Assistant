import cv2
import numpy as np

def get_colors():
    # Colors
    enemy_color1 = (202, 37, 31)
    enemy_color2 = (216, 93, 104)


    # Ranges
    h_range = 4
    s_range = 50
    v_range = 50

    # Convert Enemy 1
    enemy_rgb_color_reshape1 = np.array([[[*enemy_color1]]], dtype=np.uint8)
    enemy_hsv_color1 = cv2.cvtColor(enemy_rgb_color_reshape1, cv2.COLOR_RGB2HSV)[0][0]
    enemy_lower_hsv_color1 = np.array([enemy_hsv_color1[0] - h_range, enemy_hsv_color1[1] - s_range, enemy_hsv_color1[2] - v_range])
    enemy_upper_hsv_color1 = np.array([enemy_hsv_color1[0] + h_range, enemy_hsv_color1[1] + s_range, enemy_hsv_color1[2] + v_range])
    enemy_lower_hsv_color1 = np.clip(enemy_lower_hsv_color1, [0, 0, 0], [179, 255, 255])

    # Convert Enemy 2
    enemy_rgb_color_reshape2 = np.array([[[*enemy_color2]]], dtype=np.uint8)
    enemy_hsv_color2 = cv2.cvtColor(enemy_rgb_color_reshape2, cv2.COLOR_RGB2HSV)[0][0]
    enemy_lower_hsv_color2 = np.array([enemy_hsv_color2[0] - h_range, enemy_hsv_color2[1] - s_range, enemy_hsv_color2[2] - v_range])
    enemy_upper_hsv_color2 = np.array([enemy_hsv_color2[0] + h_range, enemy_hsv_color2[1] + s_range, enemy_hsv_color2[2] + v_range])
    enemy_lower_hsv_color2 = np.clip(enemy_lower_hsv_color2, [0, 0, 0], [179, 255, 255])

    return (enemy_lower_hsv_color1, enemy_upper_hsv_color1), (enemy_lower_hsv_color2, enemy_upper_hsv_color2)


def mask_outlines(img_hsv, w, h):
    enemy_hsv_range1, enemy_hsv_range2 = get_colors()

    enemy_mask1 = cv2.inRange(img_hsv, enemy_hsv_range1[0], enemy_hsv_range1[1])
    enemy_mask2 = cv2.inRange(img_hsv, enemy_hsv_range2[0], enemy_hsv_range2[1])

    kernel = np.ones((2,2),np.uint8)
    enemy_mask1 = cv2.dilate(enemy_mask1, kernel, iterations=3)
    enemy_mask1 = cv2.resize(enemy_mask1, (int(h / 2), int(w / 2)), interpolation=cv2.INTER_LANCZOS4)
    enemy_mask2 = cv2.dilate(enemy_mask2, kernel, iterations=2)
    enemy_mask2 = cv2.resize(enemy_mask2, (int(h / 2), int(w / 2)), interpolation=cv2.INTER_LANCZOS4)

    return enemy_mask1, enemy_mask2

img = cv2.imread("./test.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
w, h, _ = img.shape
enemy_mask1, enemy_mask2 = mask_outlines(img_hsv, w, h)
cv2.imwrite("./test_mask_1.jpg", enemy_mask1)
cv2.imwrite("./test_mask_2.jpg", enemy_mask2)