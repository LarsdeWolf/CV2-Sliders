import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch

def plot_and_save_combined_grouped_images(root_dir, save_path='combined_grouped_images_grid.png', max_groups=8):
    grouped_images = defaultdict(list)
    subfolders = sorted([os.path.join(root_dir, o) for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o))])
    
    for folder in subfolders:
        for img_name in os.listdir(folder):
            if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                if len(grouped_images) < max_groups or img_name in grouped_images:
                    full_path = os.path.join(folder, img_name)
                    grouped_images[img_name].append(full_path)
                if len(grouped_images) >= max_groups and img_name not in grouped_images:
                    break

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize each image
        transforms.ToTensor()
    ])

    all_group_tensors = []
    for img_name, paths in list(grouped_images.items())[:max_groups]:
        # Sort paths by subfolder to ensure images are ordered correctly in each row
        paths_sorted = sorted(paths, key=lambda x: subfolders.index(os.path.dirname(x)))
        images = [transform(Image.open(img).convert('RGB')) for img in paths_sorted]
        image_tensor = torch.stack(images)
        all_group_tensors.append(make_grid(image_tensor, nrow=len(paths_sorted), padding=2))

    combined_grid = make_grid(all_group_tensors, nrow=1, padding=10)

    plt.figure(figsize=(20, 3 * max_groups))
    plt.imshow(combined_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


plot_and_save_combined_grouped_images('multi_PIE_view_sorted', save_path='grouped_images_grid_view.png', max_groups=16)

