#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""Phenomenological bright-field microscope simulator for etched alpha tracks.
Generates an RGB image and a ground-truth CSV. Requires numpy, scipy, Pillow.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


def generate(output='synthetic_radon.png', truth='synthetic_radon_truth.csv',
             shape=(1056, 1408), n_tracks=46, seed=21):
    rng = np.random.default_rng(seed)
    H, W = shape
    # Smooth, slight nonuniform illumination + sensor noise.
    yy, xx = np.mgrid[:H, :W]
    background = 230 + 5 * ((xx / W) - .45) + 4 * ((yy / H) - .5)
    background += gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 75) * 75
    img = np.repeat(background[:, :, None], 3, axis=2).astype(np.float32)
    entries = []

    for k in range(n_tracks):
        x = float(rng.uniform(0, W))
        y = float(rng.uniform(0, H ))
        R = float(np.clip(rng.lognormal(np.log(17.5), .34), 7, 31))
        rim_width = float(rng.uniform(1.1, 2.5))
        contrast = float(rng.uniform(53, 125))
        focus_sigma = float(rng.choice([rng.uniform(.2, .75), rng.uniform(1.2, 3.5)], p=[.72, .28]))
        center_contrast = float(rng.uniform(-4, 9))
        pad = int(np.ceil(R + 5 * (rim_width + focus_sigma) + 5))
        x1, x2 = max(0, int(x)-pad), min(W, int(x)+pad+1)
        y1, y2 = max(0, int(y)-pad), min(H, int(y)+pad+1)
        if x2 <= x1 or y2 <= y1:
            continue
        Y, X = np.mgrid[y1:y2, x1:x2]
        radius = np.hypot(X-x, Y-y)
        ring = np.exp(-.5 * ((radius-R) / rim_width)**2).astype(np.float32)
        interior = 1.0 / (1.0 + np.exp(np.clip((radius - R) / 1.6, -30, 30)))
        # Local PSF for every track; slightly purple/blue rim as in the reference.
        ring = gaussian_filter(ring, focus_sigma)
        interior = gaussian_filter(interior.astype(np.float32), focus_sigma)
        # RGB attenuation differs by channel, creating a purple-tinted dark edge.
        for channel, factor in enumerate((.94, 1.02, .78)):
            img[y1:y2, x1:x2, channel] -= contrast * factor * ring + center_contrast * interior
        entries.append((k, x, y, R, rim_width, contrast, focus_sigma, center_contrast))

    # Debris particles: sparse small gray/purple specks, unrelated to tracks.
    for _ in range(850):
        x = int(rng.integers(W)); y = int(rng.integers(H))
        r = float(rng.uniform(.35, 1.7)); strength = float(rng.uniform(5, 42))
        margin = 5
        x1, x2 = max(0, x-margin), min(W, x+margin+1)
        y1, y2 = max(0, y-margin), min(H, y+margin+1)
        Y, X = np.mgrid[y1:y2, x1:x2]
        spot = strength * np.exp(-((X-x)**2 + (Y-y)**2)/(2*r*r))
        img[y1:y2, x1:x2] -= spot[..., None] * np.array([.85, 1., .9])
    # A few thin fiber-like scratches, gently blurred.
    fibers = np.zeros((H, W), np.float32)
    for _ in range(4):
        x0 = rng.integers(0, W); y0 = rng.integers(0, H)
        angle = rng.uniform(0, 2*np.pi)
        for t in np.linspace(0, rng.uniform(30, 90), 150):
            a = int(x0 + t*np.cos(angle) + .025*t*np.sin(t/7))
            b = int(y0 + t*np.sin(angle))
            if 0 <= a < W and 0 <= b < H: fibers[b,a] += 1
    img -= gaussian_filter(fibers, 1.25)[..., None] * 30
    img += rng.normal(0, 2.8, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img, 'RGB').save(output)
    with open(truth, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','x_px','y_px','radius_px','rim_sigma_px','contrast','focus_sigma_px','center_contrast'])
        writer.writerows(entries)
    return img, entries

# ============================================================

if __name__ == '__main__':
    F_output  = 'synthetic_radon.png'        # output image file
    F_truth   = 'synthetic_radon_truth.csv'  # output file with all truth data
    IMG_pixel = (1056, 1408)                 # image pixels
    n_tracks  = 50                           # number of tracks (circles) in the image
    seed      = 50                           # seed of random generator, keep it fix for reproducability 

    generate(output=F_output, truth=F_truth, shape=IMG_pixel,
             n_tracks=n_tracks, seed=seed)


# In[7]:


img, entries = generate(
    output='synthetic_radon.png',
    truth='synthetic_radon_truth.csv',
    shape=(1056, 1408),
    n_tracks=50,
    seed=50
)


# In[8]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()


# In[9]:


import pandas as pd

df = pd.read_csv('synthetic_radon_truth.csv')
df.head()


# In[10]:


import os

print("Current folder:", os.getcwd())
print("CSV exists:", os.path.exists('synthetic_radon_truth.csv'))
print("Files:", os.listdir())

