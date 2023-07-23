import sys
import urllib.request
import urllib.request as urllib
import zipfile


def main():
    if sys.argv[1] == 'download-models':
        print('Downloading category-related models...')

        url = 'https://zenodo.org/record/8175903/files/category_model.zip'

        filehandle, _ = urllib.urlretrieve(url)

        print('Unzipping category-related models...')
        with zipfile.ZipFile(filehandle, mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, "resources/category_model")
        print('Done!')

        print('Downloading disaster-related models...')
        url = 'https://zenodo.org/record/8175903/files/disaster_detect.zip'

        filehandle, _ = urllib.urlretrieve(url)

        print('Unzipping disaster-related models...')
        with zipfile.ZipFile(filehandle, mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, "resources/disaster_detect")

        print('Done!')

if __name__ == "__main__":
    main()