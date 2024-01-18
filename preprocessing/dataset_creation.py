import matplotlib.pyplot
import torch
import torch.utils.data
from torchvision.transforms import v2
from typing import Dict
import matplotlib as plt


class EmbryoImageDataset(torch.utils.data.Dataset):
    def __init__(self,images_dict: Dict, labels_tensor, transforms=None):
        self.images_dict = images_dict
        self.labels_tensor = labels_tensor
        self.transforms = transforms
        self.ids = labels_tensor[25, 0, :].long()
        self.ids = self.ids.tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx].item()
        image = self.images_dict[str(img_id)] / 255
        cls_label = self.labels_tensor[:25, 2:, idx]
        hole_label = self.labels_tensor[0, -1, idx]
        reg_label = self.labels_tensor[:25, :2, idx]/255
        #matplotlib.pyplot.imshow(torch.permute(image, (1,2,0)))
        #matplotlib.pyplot.show()



        #if self.transforms:
        #    image = self.transforms(image)
        #    label[:24,2:] = self.transforms(label[:24,2:])

        return image, (reg_label, cls_label, hole_label)


#images_dict = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack")
#labels_tensor = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\first_pack_labels")
#dataset = EmbryoImageDataset(images_dict, labels_tensor)
#torch.save(dataset, "C:\\Work\\EmbryoVision\\data\\torch_type\\first_dataset")



#from torch.utils.data import DataLoader
#
#dataloader = DataLoader(torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\first_dataset"), batch_size=4)
#x = 2
#y = 2
#for images, labels in dataloader:
#    print(images)
#    print(labels)
#    break

