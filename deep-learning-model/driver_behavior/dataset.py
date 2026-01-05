import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DriverBehaviorDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): True for training set, False for test set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Closed', 'Open', 'yawn', 'no_yawn']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        # Load images
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(is_train=True):
    """
    Returns the data transformations as described in the paper.
    """
    transform_list = [
        transforms.Resize((64, 64)),
    ]
    
    if is_train:
        # Paper mentions RandomHorizontalFlip with p=0.5
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # Paper also mentions rotations, brightness, contrast in text, adding for robustness
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        transform_list.append(transforms.RandomRotation(10))

    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)
