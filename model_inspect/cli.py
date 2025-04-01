import argparse
import json
import math
import struct
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import concurrent.futures

import requests
from prettytable import PrettyTable
from tqdm import tqdm

# 常量定义
SAFETENSORS_FILE = "model.safetensors"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
REVISION = "main"
HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"
HF_MIRROR_URL = "{mirror}/{repo}/resolve/{revision}/{filename}"
HEADERS = {'Range': 'bytes=0-7'}

# 数据类型到字节数的映射
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1
}

def get_file_response(repo: str, filename: str, revision: str = REVISION, range_header: str = None, verbose: int = 0, mirror: str = None) -> requests.Response:
    """获取文件响应，支持范围请求"""
    if mirror:
        url = HF_MIRROR_URL.format(mirror=mirror.rstrip('/'), repo=repo, revision=revision, filename=filename)
    else:
        url = HF_URL.format(repo=repo, revision=revision, filename=filename)
    
    headers = {'Range': range_header} if range_header else {}
    if verbose >= 1:
        print(f"Fetching URL: {url} with headers: {headers}")
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    return response

def parse_single_file(repo: str, filename: str, revision: str = REVISION, verbose: int = 0, mirror: str = None) -> dict:
    """解析单个safetensors文件头"""
    if verbose >= 1:
        print(f"Parsing single file: {filename}")
    # 获取头部长度
    response = get_file_response(repo, filename, revision, 'bytes=0-7', verbose, mirror)
    header_length = struct.unpack('<Q', response.content)[0]  # little-endian
    
    if verbose >= 1:
        print(f"Header length: {header_length}")
    
    if header_length > 25_000_000:
        raise ValueError("Header too large")
    
    # 获取实际头部内容
    range_header = f'bytes=8-{8 + header_length - 1}'
    response = get_file_response(repo, filename, revision, range_header, verbose, mirror)
    
    try:
        header = json.loads(response.content)
        if verbose >= 2:
            print(f"Parsed header: {header}")
        return header
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON header")

def parse_sharded_index(repo: str, revision: str = REVISION, verbose: int = 0, jobs: int = 1, mirror: str = None) -> Tuple[dict, dict]:
    """解析分片索引文件"""
    if verbose >= 1:
        print("Parsing sharded index file")
    # 获取索引文件
    response = get_file_response(repo, SAFETENSORS_INDEX_FILE, revision, verbose=verbose, mirror=mirror)
    index = json.loads(response.content)
    
    if verbose >= 2:
        print(f"Index content: {index}")
    
    # 获取所有分片文件头，按文件名排序
    headers = {}
    shard_files = sorted(set(index["weight_map"].values()))

    def fetch_header(filename):
        if verbose >= 1:
            print(f"Parsing shard: {filename}")
        return filename, parse_single_file(repo, filename, revision, verbose, mirror)

    with tqdm(total=len(shard_files), desc="Processing shards") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_file = {executor.submit(fetch_header, filename): filename for filename in shard_files}
            for future in concurrent.futures.as_completed(future_to_file):
                filename, header = future.result()
                headers[filename] = header
                pbar.update(1)
    
    return index, headers

def get_model_layers(repo: str, revision: str = REVISION, verbose: int = 0, jobs: int = 1, mirror: str = None) -> List[dict]:
    """获取模型层信息"""
    try:
        # 尝试获取索引文件
        if verbose >= 1:
            print("Attempting to parse sharded index")
        index, headers = parse_sharded_index(repo, revision, verbose, jobs, mirror)
        all_tensors = []
        for tensor_name, filename in index["weight_map"].items():
            header = headers[filename]
            if tensor_name in header:
                tensor_info = header[tensor_name]
                tensor_size = math.prod(tensor_info["shape"]) * DTYPE_BYTES.get(tensor_info["dtype"], 1)
                if verbose >= 1:
                    print(f"Processing tensor: {tensor_name}")
                all_tensors.append({
                    "name": tensor_name,
                    "shape": tensor_info["shape"],
                    "dtype": tensor_info["dtype"],
                    "size": tensor_size
                })
        return all_tensors
    except Exception as e:
        # 处理单个文件情况
        if verbose >= 1:
            print("Falling back to single file parsing, due to exception:", str(e))
        header = parse_single_file(repo, SAFETENSORS_FILE, revision, verbose, mirror)
        return [{
            "name": name,
            "shape": info["shape"],
            "dtype": info["dtype"],
            "size": math.prod(info["shape"]) * DTYPE_BYTES.get(info["dtype"], 1)
        } for name, info in header.items() if name != "__metadata__"]

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Model Layer Analyzer")
    parser.add_argument("model", type=str, help="Hugging Face model name (e.g. 'username/model')")
    parser.add_argument("--revision", type=str, default=REVISION, help="Model revision")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Enable verbose output (-v for process, -vv for content)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of concurrent jobs (default: 1)")
    parser.add_argument("--mirror", type=str, help="Custom Hugging Face mirror URL (e.g. 'https://hf-mirror.com')")
    args = parser.parse_args()

    try:
        layers = get_model_layers(args.model, args.revision, verbose=args.verbose, jobs=args.jobs, mirror=args.mirror)
        table = PrettyTable()
        table.field_names = ["Layer Name", "Shape", "Data Type", "Size (bytes)"]
        table.align["Layer Name"] = "l"
        table.align["Size (bytes)"] = "r"
        
        total_size = 0
        for layer in layers:
            table.add_row([
                layer["name"],
                tuple(layer["shape"]),
                layer["dtype"],
                f"{layer['size']:,}"
            ])
            total_size += layer["size"]
        
        print(table)
        print(f"\nTotal Layers: {len(layers)}")
        print(f"\nTotal Parameters Size: {total_size:,} bytes ({total_size/1024**2:.2f} MB)")
        
        # Show which URL was used
        if args.mirror:
            print(f"\nUsing mirror: {args.mirror}")
        else:
            print("\nUsing default Hugging Face URL")
        
    except requests.HTTPError as e:
        print(f"Error accessing model: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()