import cv2
import numpy as np
import pandas as pd
import imageio

def interpolate_color(color1, color2, ratio):
    return tuple(int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3))

# Read the CSV file
csv_file = 'mouse_data.csv'
df = pd.read_csv(csv_file, header=None, names=['dx', 'dy', 'time'])

# Calculate bounds of mouse movement
min_x, max_x, min_y, max_y = 0, 0, 0, 0
cursor_x, cursor_y = 0, 0

for index, row in df.iterrows():
    if index > 0 and row['dx'] != 'dx' and row['dy'] != 'dy' and row['time'] != 'time' and int(row['time']) >= 5000:
        dx, dy = row['dx'], row['dy']
        cursor_x += int(dx)
        cursor_y += int(dy)
        min_x = min(min_x, cursor_x)
        max_x = max(max_x, cursor_x)
        min_y = min(min_y, cursor_y)
        max_y = max(max_y, cursor_y)

width = max_x - min_x + 1
height = max_y - min_y + 1

# Parameters
thickness = 2
start_color = (255, 0, 0)  # Light red
end_color = (0, 0, 255)  # Light blue

# Create a black image
img = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize the cursor position
cursor_x, cursor_y = -min_x, -min_y

frames = []
durations = []
data_length = len(df)

# Draw the mouse movement path
for index, row in df.iterrows():
    if index > 0 and row['dx'] != 'dx' and row['dy'] != 'dy' and row['time'] != 'time' and int(row['time']) >= 5000:
        dx, dy = row['dx'], row['dy']
        new_x, new_y = cursor_x + int(dx), cursor_y + int(dy)
        ratio = index / data_length
        line_color = interpolate_color(start_color, end_color, ratio)

        # Calculate time difference between consecutive rows
        time_diff = 0
        if index != 1:
            time_diff = (pd.to_datetime(int(df.loc[index, 'time']), unit='ms') - pd.to_datetime(int(df.loc[index - 1, 'time']), unit='ms')).total_seconds()
        durations.append(time_diff)

        cv2.line(img, (cursor_x, cursor_y), (new_x, new_y), line_color, thickness)
        cursor_x, cursor_y = new_x, new_y
        frames.append(cv2.resize(img.copy(), (width // 1, height // 1)))

# Save as a gif
imageio.mimsave('mouse_path.gif', frames, 'GIF', duration=durations)
