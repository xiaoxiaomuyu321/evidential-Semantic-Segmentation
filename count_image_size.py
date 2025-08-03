import os
from PIL import Image

def get_image_sizes(folder_path):
    image_sizes = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                image_sizes[filename] = (width, height)
    return image_sizes

def count_unique_sizes(image_sizes):
    size_counts = {}
    for size in image_sizes.values():
        if size in size_counts:
            size_counts[size] += 1
        else:
            size_counts[size] = 1
    return size_counts

if __name__ == "__main__":
    folder_path = 'datasets/VOCdevkit/VOC2012/JPEGImages'
    image_sizes = get_image_sizes(folder_path)
    unique_sizes = count_unique_sizes(image_sizes)

    print("\n不同图片尺寸及其出现次数:")
    for size, count in unique_sizes.items():
        print(f"宽: {size[0]} 像素, 高: {size[1]} 像素 --> 出现 {count} 次")