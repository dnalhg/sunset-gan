import os
from PIL import Image
import PIL
import requests

def download_images(image_urls_file, output_dir):
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    with open(image_urls_file, "r") as image_urls:
        rows = image_urls.read().strip().split("\n")
        total = 0
        for url in rows:
            # try to download the image
            try:
                r = requests.get(url, timeout=60)
                # save the image to disk
                p = os.path.sep.join([output_dir, "{}.jpg".format(str(total).zfill(8))])
                f = open(p, "wb")
                f.write(r.content)
                f.close()
                try:
                    Image.open(p)
                    total += 1
                except PIL.UnidentifiedImageError:
                    os.remove(p)
            except:
                continue