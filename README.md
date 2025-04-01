# Model Inspect

A CLI tool to analyze Hugging Face model architectures without downloading the full model.

# Core Features

* Instant metadata analysis via the model repository
* Sharded model support (multi-safetensors files)
* Layer insights: tensor names, shapes, data types, memory estimates
* Full model metrics: parameter count and size calculation
* Mirror compatibility for downloads

## Quick Start

Install:

```bash
pip install model-inspect
```

Analyze a model with verbose output:
```bash
model-inspect Qwen/Qwen2.5-0.5B
```

Output:
```
+-------------------------------------------------+---------------+-----------+--------------+
| Layer Name                                      |     Shape     | Data Type | Size (bytes) |
+-------------------------------------------------+---------------+-----------+--------------+
| model.embed_tokens.weight                       | (151936, 896) |    BF16   |  272,269,312 |
| model.layers.0.input_layernorm.weight           |     (896,)    |    BF16   |        1,792 |
| model.layers.0.mlp.down_proj.weight             |  (896, 4864)  |    BF16   |    8,716,288 |
| model.layers.0.mlp.gate_proj.weight             |  (4864, 896)  |    BF16   |    8,716,288 |
|                                          ...                                               |
| model.layers.23.self_attn.q_proj.weight         |   (896, 896)  |    BF16   |    1,605,632 |
| model.layers.23.self_attn.v_proj.bias           |     (128,)    |    BF16   |          256 |
| model.layers.23.self_attn.v_proj.weight         |   (128, 896)  |    BF16   |      229,376 |
| model.norm.weight                               |     (896,)    |    BF16   |        1,792 |
+-------------------------------------------------+---------------+-----------+--------------+
```

Analyze a model with verbose output:
```bash
model-inspect Qwen/Qwen2.5-0.5B -v
```

Use a Hugging Face mirror:
```bash
model-inspect Qwen/Qwen2.5-0.5B --mirror https://hf-mirror.com
```

Use multiple threads for faster processing of sharded models:
```bash
model-inspect deepseek-ai/DeepSeek-V3 -j 8
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
