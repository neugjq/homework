# download_coco.py
import requests
import os
from tqdm import tqdm

def download_file(url, local_path):
    """下载大文件，显示进度条"""
    print(f"正在下载: {url}")
    print(f"保存到: {local_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, 'wb') as file, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"下载完成: {local_path}")

def main():
    # COCO数据集下载配置
    base_path = r"D:\multi_object_detection\data\coco"
    
    downloads = [
        {
            'url': 'http://images.cocodataset.org/zips/train2017.zip',
            'path': os.path.join(base_path, 'train2017.zip'),
            'size': '~19GB'
        },
        {
            'url': 'http://images.cocodataset.org/zips/val2017.zip', 
            'path': os.path.join(base_path, 'val2017.zip'),
            'size': '~1GB'
        },
        {
            'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'path': os.path.join(base_path, 'annotations_trainval2017.zip'),
            'size': '~250MB'
        }
    ]
    
    for item in downloads:
        if os.path.exists(item['path']):
            print(f"文件已存在，跳过: {os.path.basename(item['path'])}")
            continue
        
        try:
            download_file(item['url'], item['path'])
        except Exception as e:
            print(f"下载失败: {e}")
            print(f"请手动下载: {item['url']}")

if __name__ == "__main__":
    main()
