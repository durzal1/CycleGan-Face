import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, height, width, cartoon_dir, celeb_dir):
        self.height = height
        self.width = width
        self.cartoon_dir = cartoon_dir
        self.celeb_dir = celeb_dir

        self.transform = transforms.Compose([
            transforms.Resize((height, width)),  # Resize images to a consistent size
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        ])

        self.cartoon_names = [name for name in os.listdir(cartoon_dir)if not name.endswith('.csv')]
        self.celeb_names = [name for name in os.listdir(celeb_dir)]


    def __len__(self):
        return len(self.cartoon_names)

    def __getitem__(self, index):
        # Cartoon
        cartoon_name = self.cartoon_names[index]
        cartoon_path = os.path.join(self.cartoon_dir, cartoon_name)

        # Celeb
        celeb_name = self.celeb_names[index]
        celeb_path = os.path.join(self.celeb_dir, celeb_name)

        # Create tensors of (3,height,width) matrix of images
        cartoon = Image.open(cartoon_path).convert("RGB")
        celeb = Image.open(celeb_path).convert("RGB")

        # Apply transformation
        cartoon = self.transform(cartoon)
        celeb = self.transform(celeb)

        return cartoon, celeb