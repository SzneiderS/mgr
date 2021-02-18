from os.path import isfile, join
from os import listdir
import numpy
from PIL import Image
from Images.Normalization import image_to_range
import math
import itertools

images_directory = "/sdb7/seb/unprocessed_hgt/"
output_directory = "/sdb7/seb/processed_hgt/"
images = [f for f in listdir(images_directory) if isfile(join(images_directory, f))]


def hgt_filename(latitude, longitude):
    lat_half = 'N' if latitude >= 0 else 'S'
    lng_half = 'E' if longitude >= 0 else 'W'
    return "%s%02d%s%03d.hgt" % (lat_half, abs(latitude), lng_half, abs(longitude))


def load_hgt(filename):
    if isfile(join(images_directory, filename)):
        with open(join(images_directory, filename), "r") as hgt:
            elevations = numpy.fromfile(hgt, numpy.dtype(">i2"), 1201 * 1201).reshape((1201, 1201))
            return elevations
    return None


def lat_length(latitude):
    return math.cos(math.radians(latitude)) * 2.0 * math.pi * 6371.0088 * 1000


def get_elevation(latitude, longitude):
    filename = hgt_filename(latitude, longitude)
    elevations = load_hgt(filename)
    if elevations is not None:
        base_lat = int(latitude)
        base_lng = int(longitude)

        lat_row = int(abs((latitude - base_lat) * 1200))
        lng_row = int(abs((longitude - base_lng) * 1200))

        if latitude < 0:
            lat_row = 1200 - lat_row
        if longitude < 0:
            lng_row = 1200 - lng_row

        return elevations[-lat_row, lng_row].astype(int)


if __name__ == "__main__":
    img_size = (100, 100)
    kms = (100, 100)
    xkm, ykm = kms
    xkm *= 1000
    ykm *= 1000
    width, height = img_size
    points = [(65, 60)]
    for point in points:
        try:
            latitude, longitude = point
            filename = hgt_filename(latitude, longitude)
            with Image.new("F", img_size) as img:
                for y in range(height):
                    D = lat_length(0)
                    step_y = (height - y) * 360 * (ykm / height) / D
                    # step_y = (height - y) / height
                    lat_pos = latitude + step_y
                    D2 = lat_length(lat_pos)
                    for x in range(width):
                        step_x = x * 360 * (xkm / width) / D2
                        # step_x = x / width
                        lng_pos = longitude + step_x
                        elevation = get_elevation(lat_pos, lng_pos)
                        if elevation is None:
                            raise Exception()
                        img.putpixel((x, y), elevation)

                img = image_to_range(img)
                img = img.convert("RGB")
                img.save(join(output_directory, filename) + ".png")
        except Exception:
            pass
