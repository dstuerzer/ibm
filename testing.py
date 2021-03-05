import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# Generate some test data
x = np.random.randn(5)
y = np.random.randn(5)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)