import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

def create_dataloader(image_folder, batch_size, num_workers, gan_depth):
    img_size = int(np.power(2, gan_depth+1))
    dataset = dset.ImageFolder(root=image_folder,
                            transform=transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    return dataloader