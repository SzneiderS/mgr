import io
import shutil
import zipfile
from os import listdir
from os.path import isdir

import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    url = "http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm"
    output_dir = "/sdb7/seb/unprocessed_hgt2/"

    resp = requests.get(url)

    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, "html.parser")

        dem_map = soup.find("map", {
            "name": "LinkMap"
        })

        links = []
        for link in list(dem_map.findAll("area")):
            if link["href"] not in links:
                links.append(link["href"])

        for link in links:
            print("Downloading {}... ".format(link))

            link_resp = requests.get(link)

            if link_resp.status_code == 200:
                zip_file = zipfile.ZipFile(io.BytesIO(link_resp.content))

                zip_file.extractall(output_dir)

                dirs = [d + "/" for d in listdir(output_dir) if isdir(output_dir + d)]

                for directory in dirs:
                    print("Moving files from directory {}".format(directory))
                    for file in listdir(output_dir + directory):
                        shutil.move(output_dir + directory + file,
                                    output_dir + file.replace(".hgt", '').upper() + ".hgt")

                    shutil.rmtree(output_dir + directory, ignore_errors=True)

                print("Done")
            else:
                print("Failed")
