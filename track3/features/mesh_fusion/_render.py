"""Contact montages and camera-centre figures returned to the handler."""

import cv2
import numpy as np


def contact_sheet(frames, cutoff, crop=False):
    cells = []
    for index, image in sorted(frames.items()):
        if crop:
            image = image[round(image.shape[0] * 0.48):]
        cell = cv2.resize(image, (480, 270))
        colour = (0, 220, 0) if index < cutoff else (0, 80, 255)
        cv2.rectangle(cell, (0, 0), (479, 269), colour, 3)
        cv2.putText(cell, f'frame {index} | {index / 20:.2f}s', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        cells.append(cell)
    while len(cells) % 4:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])


def camera_figure(frames, object_centre, object_points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    ax = fig.add_subplot(121, projection='3d')
    points = object_points[::max(1, len(object_points) // 1500)]
    ax.scatter(*points.T, s=1, c='0.7', label='Opening stereo object')
    ax.scatter(*object_centre, s=40, c='black', label='Fixed object centre')
    flat = fig.add_subplot(122)
    flat.scatter(0, 0, c='black', label='Opening camera')
    for row in frames:
        if 'camera_centre_m' not in row:
            continue
        centre = np.array(row['camera_centre_m'])
        colour = 'tab:green' if row['accepted'] else 'tab:red'
        ax.scatter(*centre, c=colour)
        ax.plot(*np.stack([centre, object_centre]).T, c=colour, alpha=0.35)
        ax.text(*centre, str(row['frame_index']))
        flat.scatter(centre[0] * 1000, centre[2] * 1000, c=colour)
        flat.annotate(str(row['frame_index']), (centre[0] * 1000, centre[2] * 1000))
    ax.set(xlabel='Opening camera X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Camera rays; green accepted, red rejected')
    flat.set(xlabel='Camera X (mm)', ylabel='Camera Z (mm)', title='Translation around opening camera')
    flat.grid(alpha=0.3)
    fig.suptitle('F1 Stage A: video-only camera centres')
    return fig
