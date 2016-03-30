import pytumblr
import requests
import shutil

client = pytumblr.TumblrRestClient(
    'BDjcWNLhBEMzkY6UgDpmAEwThccS9wV7TPJ1AYdHaD3XBSjRGy',
    'CJwl0hc3WEOq6kAtj7QqPJtdQ4IIaJsiFRnvZiEmlAfmxJDziA',
    'Qe0WttrMtDLxPk0WZ7N3aKoxuc6lEw823BVqL2tYE2BoVN0rOr',
    'QOltqvxVUlRxK0bHc5V4lROzIQ2HE1Mow3GqZzguJhvV31tPqW',
)

print(client.info())

moviebarcode_posts = client.posts('moviebarcode', limit=5000)

post = moviebarcode_posts['posts'][0]
post2 = moviebarcode_posts['posts'][1]

print post
url = post["photos"][0]["original_size"]["url"]
url2 = post2["photos"][0]["original_size"]["url"]
post_img = requests.get(url, stream=True)

if post_img.status_code == 200:
    with open("image.jpg", 'wb') as f:
        post_img.raw.decode_content = True
        shutil.copyfileobj(post_img.raw, f)

post_img = requests.get(url2, stream=True)

if post_img.status_code == 200:
    with open("image2.jpg", 'wb') as f:
        post_img.raw.decode_content = True
        shutil.copyfileobj(post_img.raw, f)

