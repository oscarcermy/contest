import numpy as np
d = np.load("renders/depth_000.npy")
print(d.min(), d.max(), d.mean())
