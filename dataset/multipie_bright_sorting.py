import os
import shutil
from PIL import Image

def sort_images_by_brightness(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    filename_tracker = {}

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                img_path = os.path.join(root, file)

                parts = file.split('_')
                ID = parts[0]  # ID
                status = parts[1]  # status
                session = parts[2]  # session
                view = parts[3]  # view
                if int(view) != 51:
                    continue
                brightness = parts[4]  # brightness

                brightness_folder = os.path.join(destination_folder, brightness)
                if not os.path.exists(brightness_folder):
                    os.mkdir(brightness_folder)

                new_filename = f"{ID}_{status}_{session}_{view}_crop_128.jpg"
                dest_path = os.path.join(brightness_folder, new_filename)

                image = Image.open(img_path)
                image = image.convert('RGB')
                image.save(dest_path, "JPEG")

                base_filename = f"{ID}_{status}_{session}_{view}_crop_128.jpg"
                if base_filename not in filename_tracker:
                    filename_tracker[base_filename] = []
                filename_tracker[base_filename].append(brightness)

    complete_brightnesses = set(filename_tracker[next(iter(filename_tracker))])
    for key, brightnesses in filename_tracker.items():
        if set(brightnesses) != complete_brightnesses:
            for brightness in brightnesses:
                try:
                    os.remove(os.path.join(destination_folder, brightness, key))
                except OSError:
                    pass  # File may have been already deleted

source_folder = 'multi_PIE_crop_128'
destination_folder = 'multi_PIE_bright_sorted'
sort_images_by_brightness(source_folder, destination_folder)

