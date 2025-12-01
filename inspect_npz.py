import numpy as np

try:
    data = np.load('output/6CM4/batch_0000000_0000001.npz')
    print("Keys in NPZ:", list(data.keys()))
    for key in data.keys():
        print(f"Shape of {key}: {data[key].shape}")
except Exception as e:
    print(e)
