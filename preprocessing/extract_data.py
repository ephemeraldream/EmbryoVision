
import torch
import torchvision
import os
import json

class LoadData:
    def __init__(self):
        pass



    @staticmethod
    def extract_labels(filename):
        pass




    @staticmethod
    def extract_images(directory_in_str):
        directory = os.fsencode(directory_in_str)
        dir_of_images = dict()
        for file in os.listdir(directory):
            raw_filename = str(os.fsdecode(file))[:-11]
            filename_torch = directory_in_str + str(os.fsencode(file))[2:-1]
            img = torchvision.io.read_image(filename_torch)
            dir_of_images[raw_filename] = img
        torch.save(directory_in_str, "C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack")


def main():
    load = LoadData
    #load.extract_images('C:\\Work\\EmbryoVision\\data\\images\\')


if __name__ == "__main__":
    main()