import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def create_dataloader(image_folder, batch_size, num_workers):
    dataset = dset.ImageFolder(root=image_folder,
                            transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.RandomCrop(64, 0),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAutocontrast(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    return dataloader