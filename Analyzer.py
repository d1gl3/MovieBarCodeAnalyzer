import io
import json
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
    moviebarcode_posts = client.posts('moviebarcode', limit=1)['posts']

    for movie in moviebarcode_posts:
        title = movie['summary']
        img_url = movie['photos'][0]['original_size']['url']

        movies[title] = {
            "barcode_url": img_url,
            "title": title
        }

def get_picture_array_from_file(file):
    picture_array = scipy.misc.imread(file, flatten=0)
    picture_array = np.array(picture_array, dtype=np.float64) / 255
    return picture_array

def get_picture_stream_from_url(url):
    picture_stream = urllib.urlopen(url)
    picture_stream = io.BytesIO(picture_stream.read())
    return picture_stream

def transform_to_2D_numpy_array(picture_array):
    w, h, d = tuple(picture_array.shape)
    assert d == 3
    array_2D = np.reshape(picture_array, (w * h, d))
    return array_2D, w, h, d

def analyze_movie(movie, k_means, codebook_palette):

    barcode_file = get_picture_stream_from_url(movie['barcode_url'])
    barcode = get_picture_array_from_file(barcode_file)

    # Load Image and transform to a 2D numpy array.
    barcode_array, w, h, d = transform_to_2D_numpy_array(barcode)


    colors_palette = k_means.predict(barcode_array)

    count = np.bincount(colors_palette)
    ii = np.nonzero(count)
    movies[movie['title']]['colors_palette'] = zip(codebook_palette, count[ii])
    movies[movie['title']]['labels_palette'] = colors_palette
    movies[movie['title']]['whd'] = (w, h, d)
    movies[movie['title']]['codebook_palette'] = codebook_palette

    k_means, codebook = get_k_means_from_picture(barcode)

    colors_movie = k_means.predict(barcode_array)

    count = np.bincount(colors_movie)
    ii = np.nonzero(count)
    movies[movie['title']]['colors_movie'] = zip(codebook, count[ii])
    movies[movie['title']]['labels_movie'] = colors_movie
    movies[movie['title']]['codebook_movie'] = codebook

    return movie

def get_k_means_from_picture(pic):
    palette_array, w, h, d = transform_to_2D_numpy_array(pic)

    # Fitting model on a small sub-sample of the data
    palette_array_sample = shuffle(palette_array, random_state=0)[:1000]
    k_means = KMeans(n_clusters=20, random_state=0).fit(palette_array_sample)

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

    return image

if __name__ == '__main__':
    get_movie_barcodes()

    palette_pic = get_picture_array_from_file('palette.jpg')
    k_means, codebook_palette = get_k_means_from_picture(palette_pic)

    analyzed_movies = {k: analyze_movie(v, k_means, codebook_palette) for (k, v) in movies.iteritems()}

    for k, movie in analyzed_movies.iteritems():
        codebook = movie['codebook_movie']
        codebook_palette = movie['codebook_palette']
        movie_colors = movie['colors_movie'].sort(key=operator.itemgetter(1))
        movie_colors_palette = movie['colors_palette'].sort(key=operator.itemgetter(1))
        color, count = zip(*movie['colors_palette'])
        color_movie, count_movie = zip(*movie['colors_movie'])
        N = len(color)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        N_movie = len(color_movie)

        ind_movie = np.arange(N_movie)

        fig = plt.figure()
        ax1 = fig.add_subplot(324)
        ax1.set_yscale('log')
        plt.title('Color distribution palette colors (40 colors, K-Means)')
        rects1 = ax1.bar(ind, count, width)
        for i in range(len(color)):
            color_tuple = np.ceil(tuple(color[i]*255))
            hex_color = '#%02x%02x%02x' % tuple(color_tuple)
            rects1[i].set_color(hex_color)

        ax2 = fig.add_subplot(323)
        plt.axis('off')
        plt.title('Quantized image palette (40 colors, K-Means)')
        w, h, d = tuple(movie['whd'])
        plt.imshow(recreate_image(codebook_palette, movie['labels_palette'], w, h, 1))


        barcode_file = get_picture_stream_from_url(movie['barcode_url'])
        barcode_pic = get_picture_array_from_file(barcode_file)
        ax3 = fig.add_subplot(321)
        plt.axis('off')
        plt.title('Original image (40 colors, K-Means)')
        w, h, d = tuple(movie['whd'])
        plt.imshow(barcode_pic)

        ax4 = fig.add_subplot(326)
        ax4.set_yscale('log')
        plt.title('Color distribution movie colors (40 colors, K-Means)')
        rects2 = ax4.bar(ind_movie, count_movie, width)
        for i in range(len(color_movie)):
            color_tuple = np.ceil(tuple(color_movie[i] * 255))
            hex_color = '#%02x%02x%02x' % tuple(color_tuple)
            rects2[i].set_color(hex_color)

        ax5 = fig.add_subplot(325)
        plt.axis('off')
        plt.title('Quantized image movie colors (40 colors, K-Means)')
        w, h, d = tuple(movie['whd'])
        plt.imshow(recreate_image(codebook, movie['labels_movie'], w, h, 1))

    plt.show()



"""
    for k, movie in analyzed_movies.iteritems():
        colors, count = zip(*movie['colors'])
        del analyzed_movies[k]['colors']
        del analyzed_movies[k]['labels']
        converted_colors = {}
        for i in range(len(colors)):
            color_tuple = np.ceil(tuple(colors[i] * 255))
            hex_color = '#%02x%02x%02x' % tuple(color_tuple)
            converted_colors[hex_color] = count[i]

        analyzed_movies[k]['colors'] = converted_colors

#    with open("result.json", "w") as f:
#        f.write(json.dumps(analyzed_movies))
#        f.close()
"""