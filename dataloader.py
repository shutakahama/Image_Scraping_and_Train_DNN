import os
import torch.utils.data as data_utils
from torchvision import transforms
from PIL import Image


class DataLoader(data_utils.Dataset):
    def __init__(self, data, train):
        self.images = data[0]
        self.labels = data[1]

        if train:
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path = self.images[i]
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image = image.convert("RGB")
        else:
            print('cannot open this path : {0}'.format(image_path))
            raise NameError

        image = self.preprocess(image)

        return image, self.labels[i]
