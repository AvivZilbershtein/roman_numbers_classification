import os
import shutil
from PIL import Image
import numpy as np
import random
from torchvision import transforms

np.random.seed(0)
print("Your working directory is: ", os.getcwd())

base_dir = os.path.join("..", "data")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")


def redivide_train_val(train_dir, test_dir, threshold):
    classes = os.listdir(train_dir)

    for c in classes:
        train_folder_path = os.path.join(train_dir, c)
        val_folder_path = os.path.join(test_dir, c)

        images = os.listdir(train_folder_path)
        images_list = [img for img in images if img.endswith('.png')]

        total_images = len(images_list)
        train_images = int(total_images * threshold)
        random.shuffle(images_list)
        train_images_list = images_list[:train_images]
        val_images_list = images_list[train_images:]
        # move files from train to val
        for f in val_images_list:
            src = os.path.join(train_folder_path, f)
            dst = os.path.join(val_folder_path, f)
            shutil.move(src, dst)
        print(f"For class {c}, train: {len(train_images_list)}, val: {len(val_images_list)}.")


# augmenting and transforming our train files
def data_augmentation_and_transformations(train_dir):
    classes = os.listdir(train_dir)
    train_transforms = transforms.Compose([
        transforms.GaussianBlur(kernel_size=7),
        transforms.ToTensor()
    ])
    for c in classes:
        train_folder_path = os.path.join(train_dir, c)
        images = os.listdir(train_folder_path)
        images_list = [img for img in images if img.endswith('.png')]
        for i in images_list:
            img_path = os.path.join(train_folder_path, i)
            img = Image.open(img_path)
            transformed_img = train_transforms(img)
            new_img = transforms.ToPILImage()(transformed_img)
            new_path = f"{train_folder_path}/{i.split('.')[0]}augmented.png"
            new_img.save(new_path)
        print(f"Done adding augmented data to class {c}")


def count_samples(train_dir, val_dir):
    classes = os.listdir(train_dir)
    cnt = 0
    for c in classes:
        train_folder_path = os.path.join(train_dir, c)
        val_folder_path = os.path.join(val_dir, c)
        images_train = os.listdir(train_folder_path)
        cnt += len([img for img in images_train if img.endswith('.png')])
        images_val = os.listdir(val_folder_path)
        cnt += len([img for img in images_val if img.endswith('.png')])
    print(f"The number of files in both train and val folders is: {cnt}")


def get_mean_std(train_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    classes = os.listdir(train_dir)
    mean_l, std_l = [], []
    for c in classes:
        train_folder_path = os.path.join(train_dir, c)
        images = os.listdir(train_folder_path)
        images_list = [img for img in images if img.endswith('.png')]
        for img in images_list:
            img_path = os.path.join(train_folder_path, img)
            img_opened = Image.open(img_path)
            img_tr = transform(img_opened)
            mean, std = img_tr.mean([1, 2]).tolist(), img_tr.std([1, 2]).tolist()
            mean_l.append(mean)
            std_l.append(std)
    num_images = len(mean_l)
    mean = sum([mean_l[i][0] for i in range(num_images)]) / num_images
    std = sum([std_l[i][0] for i in range(num_images)]) / num_images
    return mean, std

# redivide_train_val(train_dir=train_dir, test_dir=val_dir, threshold=0.8)
# mean, std = get_mean_std(train_dir)
# print(f"Mean: {mean}. Std: {std}")
# data_augmentation_and_transformations(train_dir)
# count_samples(train_dir, val_dir)
