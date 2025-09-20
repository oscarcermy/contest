import numpy as np

value = np.load('renders/depth_000.npy', allow_pickle=True)

new = []
for i in range(960):
    for j in range(1280):
        if value[i][j]:
            new.append(1)

print(len(new))