import torch
import torchvision
import os
import json
from itertools import tee, islice, chain
from EmbryoVision.utils.tools import Tools
from typing import Any, List, Union

class LoadData:

    @staticmethod
    def previous_and_next(some_iterable):
        prevs, items, nexts = tee(some_iterable, 3)
        prevs = chain([None], prevs)
        nexts = chain(islice(nexts, 1, None), [None])
        return zip(prevs, items, nexts)

    def __init__(self):
        pass

    @staticmethod
    def extract_labels():
        holes_tensor = None
        regression_tensor = None
        classification_tensor = None
        f = open("C:\\Work\\EmbryoVision\\data\\labels\\new_labels.json")
        data = json.load(f)
        for i in range(11000):
            if 'resized' in data[i]['data']['image']:
                # only temporary.
                idx = int(data[i]['data']['image'][28:-11])
                cls_idx = torch.tensor([idx, idx, idx, idx, idx])
                reg_idx = torch.tensor([idx, idx])
                hole_idx = torch.tensor([idx])
                try:
                    annotation = data[i]['annotations'][0]['result'][0]
                    choices = annotation['value'].get('choices', [])
                    if 'U' in choices:
                        hole_tensor_to_fill = torch.ones([26,1])
                        hole_tensor_to_fill[25, :] = hole_idx
                    if 'Z' in choices:
                        zero_tensor, cls_zero_tensor, zero_holes_tensor = torch.zeros([26, 2, 1]), torch.zeros([26, 5, 1]), torch.zeros([26,1,1])
                        zero_tensor[25, :, 0], cls_zero_tensor[25, :, 0], zero_holes_tensor[25, :, 0] = reg_idx, cls_idx, hole_idx
                        zero_holes_tensor[25, :, 0] = hole_idx
                        if regression_tensor is None:
                            classification_tensor = torch.zeros([26, 5, 1])
                            regression_tensor = torch.zeros([26, 2, 1])
                            holes_tensor = torch.zeros([26,1,1])
                        else:
                            classification_tensor = torch.cat((classification_tensor, cls_zero_tensor), dim=2)
                            regression_tensor = torch.cat((regression_tensor, zero_tensor), dim=2)
                            holes_tensor = torch.cat((holes_tensor, zero_holes_tensor), dim=2)
                    else:
                        reg_tensor_to_fill = torch.zeros([26, 2])
                        cls_tensor_to_fill = torch.zeros([26, 5])
                        hole_tensor_to_fill = torch.zeros([26,1])
                        cls_tensor_to_fill[:, 4] = 1
                        count = 0
                        reg_tensor_to_fill[25, :] = reg_idx
                        cls_tensor_to_fill[25, :] = cls_idx
                        hole_tensor_to_fill[25, :] = hole_idx

                        for previous, item, next in LoadData.previous_and_next(data[i]['annotations'][0]['result']):
                            if next is None:
                                choices = item['value'].get('choices', [])
                                if "U" in choices:
                                    hole_tensor_to_fill = torch.ones([26,1])
                                    hole_tensor_to_fill[25, :] = hole_idx
                                else:
                                    to_add = torch.tensor([item['value']['x'], item['value']['y']])
                                    cls_to_add = LoadData.get_from_choice(choices)
                                    cls_to_add = torch.tensor(cls_to_add)
                                    cls_tensor_to_fill[count, :] = cls_to_add
                                    reg_tensor_to_fill[count, :] = to_add
                                    count += 1
                            elif item['id'] == next['id']:
                                continue
                            else:
                                choices = item['value'].get('choices', [])
                                if "U" in choices:
                                    hole_tensor_to_fill = torch.ones([26,1])
                                    hole_tensor_to_fill[25, :] = hole_idx
                                else:
                                    choices = item['value'].get('choices', [])
                                    cls_to_add = LoadData.get_from_choice(choices)
                                    cls_to_add = torch.tensor(cls_to_add)
                                    cls_tensor_to_fill[count, :] = cls_to_add
                                    reg_to_add = torch.tensor([item['value']['x'], item['value']['y']])
                                    reg_tensor_to_fill[count, :] = reg_to_add

                                    count += 1
                        reg_tensor_to_fill = torch.unsqueeze(reg_tensor_to_fill, 2)
                        cls_tensor_to_fill = torch.unsqueeze(cls_tensor_to_fill, 2)
                        hole_tensor_to_fill = torch.unsqueeze(hole_tensor_to_fill, 2)
                        regression_tensor = torch.cat((regression_tensor, reg_tensor_to_fill), dim=2)
                        classification_tensor = torch.cat((classification_tensor, cls_tensor_to_fill), dim=2)
                        holes_tensor = torch.cat((holes_tensor, hole_tensor_to_fill), dim=2)

                except IndexError:
                    continue
            else:
                continue

        x = 2
        result = Tools.concat_two_tensors(regression_tensor, classification_tensor)
        result = Tools.concat_two_tensors(result, holes_tensor)
        torch.save(result, "C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack_labels")

    @staticmethod
    def extract_images(directory_in_str):
        directory = os.fsencode(directory_in_str)
        dir_of_images = dict()
        for file in os.listdir(directory):
            raw_filename = str(os.fsdecode(file))[:-11]
            filename_torch = directory_in_str + str(os.fsencode(file))[2:-1]
            img = torchvision.io.read_image(filename_torch)
            dir_of_images[raw_filename] = img
        torch.save(dir_of_images, "C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack")

    @staticmethod
    def get_from_choice(choices: List) -> List:
        if not choices:
            cls_to_add = [1, 0, 0, 0, 0]
        elif 'E' in choices and 'B' not in choices:
            cls_to_add = [0, 1, 0, 0, 0]
        elif 'E' in choices and 'B' in choices:
            cls_to_add = [0, 0, 1, 0, 0]
        else:
            cls_to_add = [0, 0, 0, 1, 0]
        return cls_to_add


def main():
    load = LoadData
    load.extract_labels()
    #load.extract_images('C:\\Work\\EmbryoVision\\data\\images\\')


if __name__ == "__main__":
    main()
