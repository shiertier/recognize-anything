import os
import time
import json
import torch
import openi
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from ram import get_transform
from ram.models import ram_plus
from ram import inference_ram_openset as inference
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from ram.utils import build_openset_llm_label_embedding
from torch import nn


parser = argparse.ArgumentParser(
    description='识别万物———+————反推提示词')
parser.add_argument('--image_dir', metavar='DIR', help='path/to/dataset', default='images')
parser.add_argument('--pretrained', metavar='DIR', help='path/to/pretrained_model.pth', default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size', default=384, type=int, metavar='N', help='输入图片尺寸，默认384)')
parser.add_argument('--threshold', default=0.68, type=int, metavar='N', help='阙值，默认0.68')
parser.add_argument('--llm_tag_des',
                    metavar='DIR',
                    help='path to LLM tag descriptions',
                    default='datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json')
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

import pynvml

def get_gpu_max_memory(gpu_index=0):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if gpu_index < device_count:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        max_memory = round(meminfo.total / 1024**3) # 将字节转换为GB并四舍五入到最接近的整数
        pynvml.nvmlShutdown()
        return max_memory
    else:
        print(f"Error: GPU index {gpu_index} is out of range.")
        pynvml.nvmlShutdown()
        return None

def check_gpu_max_memory(gpu_index=0):
    max_memory = get_gpu_max_memory(gpu_index)
    lock_memory = get_gpu_lock_memory(gpu_index)
    print("你的 GPU 显存为：{} GB".format(max_memory))

    if max_memory < 4:
        print("Warning: Low GPU memory. This may cause issues.")

    print("将会限制 GPU 显存使用低于 {} GB".format(lock_memory))

def get_gpu_lock_memory(gpu_index=0):
    max_memory = get_gpu_max_memory(gpu_index)
    lock_memory = min(round(max_memory * 0.8), round(max_memory) - 1) # 取整到最接近的整数
    return lock_memory

def get_gpu_use_memory(gpu_index=0):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        use_memory = meminfo.used / 1024**3
    pynvml.nvmlShutdown()
    return use_memory

def glob_images_pathlib(dir_path):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths += list(dir_path.rglob("*" + ext))

    image_paths = list(set(image_paths))
    image_paths.sort()
    return image_paths

def download_model(user_name, model_name, model_dir):
    os_type = os.name
    if os_type == 'posix':
        save_path = '/root/.openi/token.json'
    elif os_type == 'nt':
        username = os.getlogin()
        save_path = f'C:/Users/{username}/.openi/token.json'
    else:
        raise OSError("Unsupported operating system")
    data = {
        "endpoint": "https://openi.pcl.ac.cn",
        "token": "bdcc37adfee5ca96498be4ecedff3f68e536267a"
    }
    with open(save_path, 'w') as file:
        json.dump(data, file)
    openi.model.download_model(user_name, model_name, model_dir)
    os.remove(save_path)

def check_model_exist():
    pretrained_path = Path(args.pretrained)
    if not os.path.exists(pretrained_path):
        parent_dir = os.path.dirname(os.path.abspath(pretrained_path))
        print(f"模型 {pretrained_path} 不存在，将会下载到 {parent_dir}")
        model_name = "ram"
        download_model("shiertier/12T_comfyui", model_name, parent_dir)
        os.rename(os.path.join(parent_dir, model_name, "ram_plus_swin_large_14m.pth"), parent_dir)
        os.rmdir(os.path.join(parent_dir, model_name))

def transform_and_inference(image_path, model, transform, device):
    try:
        image_res = transform(Image.open(image_path)).unsqueeze(0).to(device)
        res = inference(image_res, model)
        tag = res.replace(" | ", ", ")
        del image_res
        return tag
    except Exception as e:
        print(f"错误！{image_path}: {e}")
        return None

def transform_and_inference_2(image_path, model, transform, device):
    try:
        image_res = transform(Image.open(image_path)).unsqueeze(0).to(device)
        res = inference(image_res, model)
        tag = res.replace(" | ", ", ")
        del image_res
        return tag
    except Exception as e:
        print(f"错误！{image_path}: {e}")
        return None

def process_single_image(image_path, model, transform, device, houzhui):
    txt_path = os.path.splitext(image_path)[0] + '.' + houzhui
    if os.path.exists(txt_path):
        print(f"{image_path}已有标签.")
        return
    if houzhui == "txt1":
        tag = transform_and_inference(image_path, model, transform, device)
    else :
        tag = transform_and_inference_2(image_path, model, transform, device)
    if tag is not None:
        with open(txt_path, 'w') as txt_file:
            txt_file.write(tag)

def process_images_in_directory(dir_path, model, transform, device, lock_memory, houzhui):
    image_paths = glob_images_pathlib(dir_path)
    with ThreadPoolExecutor() as executor:
        try:
            for image_path in image_paths:
                use_memory = get_gpu_use_memory()
                if use_memory < lock_memory:
                    executor.submit(process_single_image, image_path, model, transform, device, houzhui)
                else:
                    time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            executor.shutdown(wait=True)  # Wait for all tasks to complete
            del model  # Clean up the model
    print("所有图像已提交处理。")

def model_1(image_dir, transform, device, lock_memory):
    print("正在将模型1装载到cuda...")
    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l', threshold=args.threshold)
    model.eval()
    model = model.to(device)
    print("开始反推1")
    process_images_in_directory(image_dir, model, transform, device, lock_memory, "txt1")

def model_2(image_dir, transform, device, lock_memory):
    print("正在将模型2装载到cuda...")
    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l', threshold=args.threshold)
    print('Building tag embedding:')
    with open(args.llm_tag_des, 'rb') as fo:
        llm_tag_des = json.load(fo)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)
    model.tag_list = np.array(openset_categories)
    model.label_embed = nn.Parameter(openset_label_embedding.float())
    model.num_class = len(openset_categories)
    model.class_threshold = torch.ones(model.num_class) * 0.5
    model.eval()
    model = model.to(device)
    print("开始反推2")
    process_images_in_directory(image_dir, model, transform, device, lock_memory, "txt2")   

if __name__ == "__main__":
    args = parser.parse_args()
    print("检查显卡")
    check_gpu_max_memory(gpu_index=0)
    lock_memory = get_gpu_lock_memory(gpu_index=0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("加载cuda设备成功")
    else:
        print("Error: CUDA is not available. Exiting the program.")
        exit(1)
    transform = get_transform(image_size=args.image_size)
    print("配置transform成功")
    check_model_exist()
    image_dir = Path(args.image_dir)
    model_1(image_dir, transform, device, lock_memory)
    #model_2(image_dir, transform, device, lock_memory)
