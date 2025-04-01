# Model Inspect

A command-line tool to analyze and inspect Hugging Face model architectures directly from the repository without downloading the entire model.

## Overview

Model Inspect helps you understand the structure of large language models by fetching and analyzing only the model's metadata from the Hugging Face model repository. It extracts information about model layers, their shapes, data types, and sizes without requiring a full model download.

## Features

- Inspects model architecture from Hugging Face repositories
- Works with sharded models (multiple safetensors files)
- Shows tensor names, shapes, data types, and memory sizes
- Calculates total parameter count and model size
- Supports concurrent processing for faster analysis of sharded models
- Compatible with Hugging Face mirrors

## Installation

```bash
pip install model-inspect
```

Or install from source:

```bash
git clone https://github.com/simpx/model-inspect.git
cd model-inspect
pip install -e .
```

## Usage

Basic usage:

```bash
model-inspect username/model-name
```

Example:

```bash
model-inspect meta-llama/Llama-2-7b-hf
```

### Command-line Options

```
usage: model-inspect [-h] [--revision REVISION] [-v] [-j JOBS] [--mirror MIRROR] [--timeout TIMEOUT] [--retries RETRIES] [--backoff BACKOFF] model

Hugging Face Model Layer Analyzer

positional arguments:
  model                 Hugging Face model name (e.g. 'username/model')

optional arguments:
  -h, --help            show this help message and exit
  --revision REVISION   Model revision (default: main)
  -v, --verbose         Enable verbose output (-v for process, -vv for content)
  -j JOBS, --jobs JOBS  Number of concurrent jobs (default: 1)
  --mirror MIRROR       Custom Hugging Face mirror URL (e.g. 'https://hf-mirror.com')
  --timeout TIMEOUT     Request timeout in seconds (default: 30)
  --retries RETRIES     Number of retry attempts (default: 3)
  --backoff BACKOFF     Backoff factor for retries (default: 1)
```

### Examples

Analyze a model with verbose output:
```bash
model-inspect meta-llama/Llama-2-7b-hf -v
```

Use a Hugging Face mirror:
```bash
model-inspect meta-llama/Llama-2-7b-hf --mirror https://hf-mirror.com
```

Use multiple threads for faster processing of sharded models:
```bash
model-inspect meta-llama/Llama-2-70b-hf -j 8
```

#### Example Output

Here's an example of running `model-inspect` on DeepSeek-V3:

```bash
model-inspect deepseek-ai/DeepSeek-V3
```

Output:
```
+---------------------------------------------------------------+----------------+-----------+---------------+
| Layer Name                                                    |     Shape      | Data Type |  Size (bytes) |
+---------------------------------------------------------------+----------------+-----------+---------------+
| model.embed_tokens.weight                                     | (129280, 7168) |    BF16   | 1,853,358,080 |
| model.layers.0.self_attn.q_a_proj.weight                      |  (1536, 7168)  |  F8_E4M3  |    11,010,048 |
| model.layers.0.self_attn.q_a_proj.weight_scale_inv            |    (12, 56)    |    F32    |         2,688 |
| model.layers.0.self_attn.q_a_layernorm.weight                 |    (1536,)     |    BF16   |         3,072 |
| model.layers.0.self_attn.q_b_proj.weight                      | (24576, 1536)  |  F8_E4M3  |    37,748,736 |
| model.layers.0.self_attn.q_b_proj.weight_scale_inv            |   (192, 12)    |    F32    |         9,216 |
| model.layers.0.self_attn.kv_a_proj_with_mqa.weight            |  (576, 7168)   |  F8_E4M3  |     4,128,768 |
| model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv  |    (5, 56)     |    F32    |         1,120 |
| model.layers.0.self_attn.kv_a_layernorm.weight                |     (512,)     |    BF16   |         1,024 |
| model.layers.0.self_attn.kv_b_proj.weight                     |  (32768, 512)  |  F8_E4M3  |    16,777,216 |
| model.layers.0.self_attn.kv_b_proj.weight_scale_inv           |    (256, 4)    |    F32    |         4,096 |
| model.layers.0.self_attn.o_proj.weight                        | (7168, 16384)  |  F8_E4M3  |   117,440,512 |
| model.layers.0.self_attn.o_proj.weight_scale_inv              |   (56, 128)    |    F32    |        28,672 |
| model.layers.0.mlp.gate_proj.weight                           | (18432, 7168)  |  F8_E4M3  |   132,120,576 |
| model.layers.0.mlp.gate_proj.weight_scale_inv                 |   (144, 56)    |    F32    |        32,256 |
| model.layers.0.mlp.up_proj.weight                             | (18432, 7168)  |  F8_E4M3  |   132,120,576 |
| model.layers.0.mlp.up_proj.weight_scale_inv                   |   (144, 56)    |    F32    |        32,256 |
| model.layers.0.mlp.down_proj.weight                           | (7168, 18432)  |  F8_E4M3  |   132,120,576 |
| model.layers.0.mlp.down_proj.weight_scale_inv                 |   (56, 144)    |    F32    |        32,256 |
| model.layers.0.input_layernorm.weight                         |    (7168,)     |    BF16   |        14,336 |
| model.layers.0.post_attention_layernorm.weight                |    (7168,)     |    BF16   |        14,336 |
| model.layers.1.self_attn.q_a_proj.weight                      |  (1536, 7168)  |  F8_E4M3  |    11,010,048 |
| model.layers.1.self_attn.q_a_proj.weight_scale_inv            |    (12, 56)    |    F32    |         2,688 |
| ... |
| model.layers.61.input_layernorm.weight                        |    (7168,)     |    BF16   |        14,336 |
| model.layers.61.post_attention_layernorm.weight               |    (7168,)     |    BF16   |        14,336 |
| model.layers.61.embed_tokens.weight                           | (129280, 7168) |    BF16   | 1,853,358,080 |
| model.layers.61.enorm.weight                                  |    (7168,)     |    BF16   |        14,336 |
| model.layers.61.hnorm.weight                                  |    (7168,)     |    BF16   |        14,336 |
| model.layers.61.eh_proj.weight                                | (7168, 14336)  |    BF16   |   205,520,896 |
| model.layers.61.shared_head.norm.weight                       |    (7168,)     |    BF16   |        14,336 |
| model.layers.61.shared_head.head.weight                       | (129280, 7168) |    BF16   | 1,853,358,080 |
+---------------------------------------------------------------+----------------+-----------+---------------+

Total Layers: 91991

Total Parameters Size: 688,574,839,360 bytes (656676.14 MB)
```

## How It Works

Model Inspect works by:

1. Fetching the model's safetensors index file (if available, for sharded models)
2. Reading only the headers of the safetensors files to extract tensor metadata
3. Computing tensor sizes based on shapes and data types
4. Presenting a summary table of all tensors and their properties

This approach avoids downloading the actual model weights, making it much faster and resource-efficient than downloading the entire model.

## Requirements

- Python 3.7+
- requests
- prettytable
- tqdm

## License

[MIT License](LICENSE)