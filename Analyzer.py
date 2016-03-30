import io
import urllib
import operator

import matplotlib.pyplot as plt
import numpy as np
import pytumblr
import scipy.misc
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

client = pytumblr.TumblrRestClient(
    'BDjcWNLhBEMzkY6UgDpmAEwThccS9wV7TPJ1AYdHaD3XBSjRGy',
    'CJwl0hc3WEOq6kAtj7QqPJtdQ4IIaJsiFRnvZiEmlAfmxJDziA',
    'Qe0WttrMtDLxPk0WZ7N3aKoxuc6lEw823BVqL2tYE2BoVN0rOr',
    'QOltqvxVUlRxK0bHc5V4lROzIQ2HE1Mow3GqZzguJhvV31tPqW',
)

print(client.info())
movies = {}

def get_movie_barcodes():
    moviebarcode_posts = client.posts('moviebarcode', limit=3)['posts']

    for movie in moviebarcode_posts:
        title = movie['summary']
        img_url = movie['photos'][0]['original_size']['url']

        movies[title] = {
            "barcode_url": img_url,
            "title": title
        }

    movies['BluesBrothers'] = {
        "barcode_url": "image4.jpg",
        "title": "BluesBrothers"
    }

    movies['PusteBlume'] = {
        "barcode_url": "image3.jpg",
        "title": "PusteBlume"
    }

def analyze_movie(movie, k_means, codebook):
    if movie['title'] in ("BluesBrothers", "PusteBlume"):
        barcode_file = movie['barcode_url']
    else:
        barcode_req = urllib.urlopen(movie['barcode_url'])

        barcode_file = io.BytesIO(barcode_req.read())

    barcode = scipy.misc.imread(barcode_file, flatten=0)
    barcode = np.array(barcode, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(barcode.shape)
    assert d == 3
    barcode_array = np.reshape(barcode, (w * h, d))

    colors = k_means.predict(barcode_array)

    count = np.bincount(colors)
    ii = np.nonzero(count)
    movies[movie['title']]['colors'] = zip(codebook, count[ii])
    movies[movie['title']]['labels'] = colors
    movies[movie['title']]['whd'] = (w,h,d)

    return movie


def get_k_means_from_128_palette():
    palette = scipy.misc.imread('palette.jpg', flatten=0)
    palette = np.array(palette, dtype=np.float64) / 255
    w, h, d = tuple(palette.shape)
    assert d == 3
    palette_array = np.reshape(palette, (w * h, d))

    # Fitting model on a small sub-sample of the data
    palette_array_sample = shuffle(palette_array, random_state=0)[:1000]
    k_means = KMeans(n_clusters=128, random_state=0).fit(palette_array_sample)

    codebook = k_means.cluster_centers_

    return k_means, codebook

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

if __name__ == '__main__':
    get_movie_barcodes()
    k_means, codebook = get_k_means_from_128_palette()

    analyzed_movies = {k: analyze_movie(v, k_means, codebook) for (k, v) in movies.iteritems()}

    for k, movie in analyzed_movies.iteritems():
        movie_colors = movie['colors'].sort(key=operator.itemgetter(1))
        color, count = zip(*movie['colors'])
        N = len(color)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yscale('log')
        rects1 = ax1.bar(ind, count, width)
        for i in range(len(color)):
            color_tuple = np.ceil(tuple(color[i]*255))
            hex_color = '#%02x%02x%02x' % tuple(color_tuple)
            rects1[i].set_color(hex_color)

        ax2 = fig.add_subplot(212)
        plt.axis('off')
        plt.title('Quantized image (256 colors, K-Means)')
        w, h, d = tuple(movie['whd'])
        plt.imshow(recreate_image(k_means.cluster_centers_, movie['labels'], w, h, 1))

        plt.show()

    print ""
