# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause
import scipy.misc

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_files
from sklearn.utils import shuffle
from time import time

n_colors = 128

colors = {
    "img1": {},
    "img2": {}
    }

# Load the Summer Palace photo
image1 = scipy.misc.imread("image.jpg", flatten=0)
image2 = scipy.misc.imread("image2.jpg", flatten=0)


palette = scipy.misc.imread("palette.jpg", flatten=0)
palette = np.array(palette, dtype=np.float64) / 255

w, h, d = palette_shape = tuple(palette.shape)
assert d == 3
palette_array = np.reshape(palette, (w * h, d))

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1]
image1 = np.array(image1, dtype=np.float64) / 255
image2 = np.array(image2, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image1.shape)
assert d == 3
image_array1 = np.reshape(image1, (w * h, d))

w1, h1, d1 = original_shape2 = tuple(image1.shape)
assert d == 3
image_array2 = np.reshape(image2, (w1 * h1, d1))

image_array = np.concatenate((image_array1, image_array2), axis=0)

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(palette_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels1 = kmeans.predict(image_array1)
labels2 = kmeans.predict(image_array2)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array1,
                                          axis=0)
labels_random2 = pairwise_distances_argmin(codebook_random,
                                          image_array2,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h, img_id=None):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1

    y = np.bincount(labels)
    ii = np.nonzero(y)
    print zip(codebook, y[ii])

    return image

# Display all results, alongside original image
plt.figure()
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(image1)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels1, w, h, 1))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))

# Display all results, alongside original image
plt.figure(4)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image2 (96,615 colors)')
plt.imshow(image2)

plt.figure(5)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image2 (4 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels2, w1, h1, 2))

plt.figure(6)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image 64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random2, w1, h1))
plt.show()

print ""