import torch
import torchvision
import os
import json


class LoadData:
    def __init__(self):
        pass

    @staticmethod
    def extract_labels():
        regression_tensor = torch.tensor([])
        f = open("C:\\Work\\EmbryoVision\\data\\labels\\raw_labels.json")
        data = json.load(f)
        for i in range(1000):
            z = data[i]
            if 'choices' in data[i]['annotations'][0]['result'][0]['value'].keys():
                x = data[i]['annotations'][0]['result'][0]['value']
                if data[i]['annotations'][0]['result'][0]['value']['choices'][0] == 'Z':
                    zero_tensor = torch.zeros([26,2])
                    added = torch.tensor([data[i]['id'], data[i]['id']])
                    zero_tensor[25,:] = added
                    regression_tensor = torch.cat((regression_tensor,zero_tensor))

            else:
                added = torch.tensor([data[i]['id'], data[i]['id']])
                tensor_to_fill = torch.zeros([26,2])
                count = 0
                tensor_to_fill[25,:] = added
                for labs in data[i]['annotations'][0]['result']:
                    if 'choices' not in labs['value'].keys():
                        to_add = torch.tensor([labs['value']['x'], labs['value']['y']])
                        tensor_to_fill[count,:] = to_add
                        count += 1
                regression_tensor = torch.cat((regression_tensor,tensor_to_fill))

        torch.save(regression_tensor, "C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack_labels")

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
    load.extract_labels()
    # load.extract_images('C:\\Work\\EmbryoVision\\data\\images\\')


if __name__ == "__main__":
    main()
