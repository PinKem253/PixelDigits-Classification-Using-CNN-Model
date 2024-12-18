
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir:str,
    test_dir:str,
    batch_size:int,
    num_workers= NUM_WORKERS,
):
  train_transform = transforms.Compose([
      transforms.RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.9,1.1)),
      transforms.ToTensor(),
  ])

  test_transform = transforms.Compose([
      transforms.ToTensor(),
  ])


  train_data = datasets.MNIST(
      root=train_dir,
      train=True,
      download= True,
      transform = train_transform,
  )

  test_data = datasets.MNIST(
      root=test_dir,
      train=False,
      download=True,
      transform = test_transform,
  )


  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers= num_workers,
      pin_memory= True,
  )

  test_dataloader = DataLoader(
      test_data,
      batch_size= batch_size,
      shuffle=False,
      num_workers = num_workers,
      pin_memory = True,
  )

  return train_dataloader,test_dataloader,class_names
