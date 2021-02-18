import math
import numpy
import random

from PIL import Image

from os.path import isdir, isfile, join
from os import listdir

from Images.Normalization import normalize
import torchvision.transforms as transforms

import tarfile

from abc import ABC, abstractmethod

processed = 'processed_hgt/'

EARTH_RADIUS = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS * 1000.0


class HGTFileNotFound(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.message = message
        self.errors = errors

    def __str__(self):
        return "%s (%s)" % (self.message, self.errors)


class BaseProcessor(ABC):
    def __init__(self):
        self.dataset_min = None
        self.dataset_max = None

        self.use_dataset_min_max = False
        self.min_override = None
        self.max_override = None

        self.max_loaded = None

        self.loaded = {}

        self.filename_blacklist = set()

    @abstractmethod
    def _get_minmax(self):
        raise NotImplementedError()

    @staticmethod
    def hgt_filename(latitude, longitude):
        lat_half = 'N' if latitude >= 0 else 'S'
        lng_half = 'E' if longitude >= 0 else 'W'
        return "%s%02d%s%03d.hgt" % (lat_half, abs(latitude), lng_half, abs(longitude))

    @staticmethod
    def _lat_length(latitude):
        return math.cos(math.radians(latitude)) * 2.0 * math.pi * EARTH_RADIUS * 1000

    @staticmethod
    def generate_histogram(img, bins=255):
        tensorize = transforms.ToTensor()
        img_tensor = tensorize(img)
        histogram = img_tensor.histc(bins=bins, min=0, max=1)
        histogram /= histogram.sum()
        return histogram

    def min(self):
        if self.dataset_min is None:
            self._get_minmax()
        return self.dataset_min

    def max(self):
        if self.dataset_max is None:
            self._get_minmax()
        return self.dataset_max

    def get_elevation(self, latitude, longitude):
        filename = self.hgt_filename(latitude, longitude)
        if filename in self.filename_blacklist:
            raise HGTFileNotFound("HGT file not found", filename)
        try:
            if filename in self.loaded:
                elevations = self.loaded[filename]
            else:
                elevations = self.get_elevation_data(filename)
                if elevations is None:
                    if filename not in self.filename_blacklist:
                        self.filename_blacklist.add(filename)
                    raise HGTFileNotFound("HGT file not found", filename)

                maximum = numpy.amax(elevations)
                minimum = numpy.amin(elevations)
                if minimum == -32768 or (maximum - minimum) < 25:
                    if filename not in self.filename_blacklist:
                        self.filename_blacklist.add(filename)
                    raise HGTFileNotFound("HGT file not found", filename)
                self.loaded[filename] = elevations

                if self.max_loaded is not None:
                    loaded_length = len(self.loaded)
                    if self.max_loaded < loaded_length:
                        del self.loaded[list(self.loaded.keys())[random.randint(0, loaded_length - 1)]]

            base_lat = int(latitude)
            base_lng = int(longitude)

            lat_row = int(abs((latitude - base_lat) * 1200))
            lng_row = int(abs((longitude - base_lng) * 1200))

            if latitude < 0:
                lat_row = 1200 - lat_row
            if longitude < 0:
                lng_row = 1200 - lng_row

            return elevations[-lat_row, lng_row].astype(int)
        except KeyError:
            if filename not in self.filename_blacklist:
                self.filename_blacklist.add(filename)
            raise HGTFileNotFound("HGT file not found", filename)

    @abstractmethod
    def get_elevation_data(self, filename):
        raise NotImplementedError()

    def generate_image(self, geo_pos, img_size, save_dir, meters_per_pixel=(1, 1)):
        maximum = None
        minimum = None
        if self.use_dataset_min_max and (self.min_override is None and self.max_override is None):
            maximum = self.max()
            minimum = self.min()
        if self.min_override is not None and self.max_override is not None:
            maximum = self.max_override
            minimum = self.min_override
        try:
            latitude, longitude = geo_pos
            width, height = img_size
            with Image.new("F", img_size) as img:
                for y in range(height):
                    lat_pos = latitude + (height - y) * 360.0 * meters_per_pixel[0] / self._lat_length(0)
                    lat_length = self._lat_length(lat_pos)
                    for x in range(width):
                        lng_pos = longitude + x * 360.0 * meters_per_pixel[1] / lat_length
                        elevation = self.get_elevation(lat_pos, lng_pos)
                        img.putpixel((x, y), elevation)

                img = normalize(img, minimum, maximum)
                histogram = self.generate_histogram(img, 32)
                if histogram.max() > random.normalvariate(0.5, 0.05/3.0):
                    return False
                img = img.convert("RGB")
                img.save(save_dir + "(%s)(%s)%dkmx%dkm.png" % (
                    "{0:.2f}".format(latitude).replace('.', '_'), "{0:.2f}".format(longitude).replace('.', '_'),
                    width * meters_per_pixel[0] / 1000.0, height * meters_per_pixel[1] / 1000.0))
                return True
        except HGTFileNotFound:
            pass
        return False

    def generate_image_kms(self, geo_pos, img_size, kms, save_dir):
        width, height = img_size
        xkm, ykm = kms
        xkm *= 1000
        ykm *= 1000
        return self.generate_image(geo_pos, img_size, save_dir, (xkm / width, ykm / height))


class ArchiveDatasetProcessor(BaseProcessor):
    def get_elevation_data(self, filename):
        hgt = self.archive.extractfile(self.archive_members[filename].name)
        if hgt:
            elevations = numpy.frombuffer(hgt.read(), numpy.dtype(">i2"), 1201 * 1201).reshape((1201, 1201))
            return elevations
        return None

    def __init__(self, archive):
        super(ArchiveDatasetProcessor, self).__init__()
        self.archive = None
        self.archive_members = None

        self._load_archive_members(archive)

    def _load_archive_members(self, archive_name):
        self.archive = tarfile.open(archive_name, "r")
        self.archive_members = {x.name.replace("unprocessed_hgt/", ''): x for x in self.archive.getmembers()}

    def _get_minmax(self):
        if self.archive is not None:
            print("Finding min-max")

            ming = 2**15 - 1
            maxg = -(2**15)

            i = 0
            total = len(self.archive_members)

            for member in self.archive_members:
                if i % 250 == 0:
                    print("Progress: {:.3f}%".format(100 * i / total))
                elevations = self.get_elevation_data(member)
                if elevations is not None:
                    min_file = numpy.amin(elevations)
                    max_file = numpy.amax(elevations)
                    if ming > min_file != -32768:
                        ming = min_file
                    if maxg < max_file:
                        maxg = max_file
                i += 1

            self.dataset_min = ming
            self.dataset_max = maxg

            print("Min: {}, Max: {}".format(ming, maxg))

            return ming, maxg


class DirectoryDatasetProcessor(BaseProcessor):

    def __init__(self, directory):
        assert isdir(directory), "Dataset {} directory must exist".format(directory)

        super(DirectoryDatasetProcessor, self).__init__()

        self.directory = directory

    def _get_minmax(self):
        print("Finding min-max")

        ming = 2 ** 15 - 1
        maxg = -(2 ** 15)

        i = 0

        files = [join(self.directory, f) for f in listdir(self.directory) if isfile(join(self.directory, f))]

        total = len(files)

        for file in files:
            if i % 250 == 0:
                print("Progress: {:.3f}%".format(100 * i / total))
            elevations = self.get_elevation_data(join(self.directory, file))
            if elevations is not None:
                min_file = numpy.amin(elevations)
                max_file = numpy.amax(elevations)
                if ming > min_file != -32768:
                    ming = min_file
                if maxg < max_file:
                    maxg = max_file
            i += 1

        self.dataset_min = ming
        self.dataset_max = maxg

        print("Min: {}, Max: {}".format(ming, maxg))

        return ming, maxg

    def get_elevation_data(self, filename):
        if isfile(join(self.directory, filename)):
            with open(join(self.directory, filename), "r") as hgt:
                elevations = numpy.fromfile(hgt, numpy.dtype(">i2"), 1201 * 1201).reshape((1201, 1201))
                return elevations
        return None


images_generated = 0
images_required_per_size = 1000
scale_multiplier = 10
sizes = [(10 * scale_multiplier, 10 * scale_multiplier)]

processor = DirectoryDatasetProcessor("/sdb7/seb/unprocessed_hgt/")
# processor = ArchiveDatasetProcessor(".unprocessed_hgt.tar.gz")
# processor.use_dataset_min_max = True
# processor.min_override = -1130
# processor.max_override = 8840
processor.max_loaded = 1000

# for size in sizes:
#     for lat in range(-60, 61):
#         for lng in range(-180, 181):
#             processor.generate_image_kms((lat, lng), (500, 500), size, "/sdb7/seb/hgt_images/")
#             print("%s" % ((lat, lng),))


for _ in range(0, 10):
    for size in sizes:
        images_generated = 0
        while images_generated < images_required_per_size:
            lat = random.uniform(-60, 60)
            lng = random.uniform(-180, 180)
            if processor.generate_image_kms((lat, lng), (512, 512), size, "/sdb7/seb/hgt_images/"):
                images_generated += 1
                print("%s, %d/%d" % (size, images_generated, images_required_per_size))
