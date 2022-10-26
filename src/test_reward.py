#!/usr/bin/env python
import numpy as np

a = 0.4 * np.ones(360).astype(np.float32)
a = np.abs(a - 0.4)

curr_lidar = a[44:135]
indices = np.expand_dims(np.arange(curr_lidar.size), axis=0)

mu  = np.mean(indices, keepdims=True)
sigma = np.std(indices, keepdims=True)

wts = (0.4 * np.exp(-(0.5) * ((indices - mu) / (sigma)) ** 2))
weighted_lidar = curr_lidar * wts
reward = -1 * np.mean(weighted_lidar)


print(reward)
