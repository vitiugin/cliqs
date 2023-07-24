import os
import sys
import urllib.request
import urllib.request as urllib
import zipfile

class_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources/disaster_detect')
sum_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources/category_model')
queries_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


def main():
    if sys.argv[1] == 'download-models':

        print('Downloading category-related models...')

        url = 'https://zenodo.org/record/8177143/files/category_model.zip'

        filehandle, _ = urllib.urlretrieve(url)

        print('Unzipping category-related models...')
        with zipfile.ZipFile(filehandle, mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, sum_output_dir)
        print('Done!')

        print('Downloading disaster-related models...')
        url = 'https://zenodo.org/record/8177143/files/disaster_detect.zip'

        filehandle, _ = urllib.urlretrieve(url)

        print('Unzipping disaster-related models...')
        with zipfile.ZipFile(filehandle, mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, class_output_dir)

        print('Done!')


        print('Downloading queries...')
        url = 'https://zenodo.org/record/8177143/files/queries.json.zip'

        filehandle, _ = urllib.urlretrieve(url)

        print('Unzipping queries...')
        with zipfile.ZipFile(filehandle, mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, queries_output_dir)

        print('Done!')

if __name__ == "__main__":
    main()