# DGX Inference Server

A comprehensive multi-modal inference serving platform optimized for NVIDIA DGX systems. Supports text embeddings, CLIP vision-language models, computer vision, and document reranking with TensorRT optimization and Triton Inference Server.

## 🏗️ Architecture

```
dgx-inference-server/
├── models/                    # 🔹 Model definitions (PyTorch source)
│   ├── text_embedding/
│   │   ├── model.py
│   │   ├── tokenizer.py
│   │   ├── config.yaml
│   │   └── weights/
│   ├── clip/
│   │   ├── image_encoder.py
│   │   ├── text_encoder.py
│   │   ├── preprocess.py
│   │   └── weights/
│   ├── vision/
│   │   ├── model.py
│   │   ├── preprocess.py
│   │   └── weights/
│   ├── reranker/
│   │   ├── model.py
│   │   └── weights/
│   └── common/
│       ├── base.py
│       └── pooling.py
│
├── export/                    # 🔹 ONNX export scripts  
│   ├── text_embedding/
│   │   └── export_onnx.py
│   ├── clip/
│   │   ├── export_image_encoder.py
│   │   └── export_text_encoder.py
│   ├── vision/
│   │   └── export_onnx.py
│   └── reranker/
│       └── export_onnx.py
│
├── onnx_models/              # 🔹 Generated ONNX files (versioned)
│   ├── text_embedding/1/model.onnx
│   ├── clip_image/1/model.onnx  
│   ├── clip_text/1/model.onnx
│   ├── vision/1/model.onnx
│   └── reranker/1/model.onnx
│
├── tensorrt/                 # 🔹 TensorRT optimization
│   ├── build_engine.py
│   ├── configs/
│   │   ├── text_embedding.yaml
│   │   ├── clip_image.yaml
│   │   ├── vision.yaml
│   │   └── reranker.yaml
│   └── engines/              # Hardware-bound artifacts
│       ├── text_embedding/1/model.plan
│       ├── clip_image/1/model.plan
│       └── vision/1/model.plan
│
├── triton_serve/             # 🔹 Triton model repository
│   ├── text_embedding/
│   │   ├── config.pbtxt
│   │   └── 1/model.plan
│   ├── clip_image/
│   ├── clip_text/
│   ├── vision/
│   └── reranker/
│
├── server/                   # 🔹 Custom router API
│   ├── main.py
│   ├── router.py
│   └── schemas.py
│
├── docker/                   # 🔹 Container configurations
│   ├── Dockerfile.export
│   ├── Dockerfile.trt
│   └── Dockerfile.server
│
├── benchmarks/               # 🔹 Performance testing
│   ├── latency_test.py
│   └── throughput_test.py
│
└── README.md
```

## ⚙️ Installation

### Prerequisites
- **Python 3.11** (recommended for GPU PyTorch compatibility)
- **NVIDIA GPU** with CUDA ≥11.8
- **Git** for cloning repository

### 1. Setup Python Environment

```bash
# Clone repository
git clone <repository-url>
cd inference_server

# Create Python 3.11 virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate  # Linux/Mac
# OR
.venv311\Scripts\activate     # Windows
```

### 2. Install Dependencies

#### Option A: Automated Installation (Recommended)
```bash
# Run the installation script
chmod +x install_requirements.sh
./install_requirements.sh
```

#### Option B: Manual Installation
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install by category
pip install -r requirements-export.txt    # AI/ML packages
pip install -r requirements-server.txt    # Web server
pip install -r requirements-tensorrt.txt  # TensorRT (optional)

# OR install base development packages only
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import onnx; print(f'ONNX: {onnx.__version__}')"
```

### 4. TensorRT Setup (Optional)

For maximum performance, install NVIDIA TensorRT:

```bash
# Install TensorRT (requires NVIDIA developer account)
# Visit: https://developer.nvidia.com/tensorrt

# Or use pip (limited functionality)
pip install tensorrt
```

### 📁 Requirements Files

- **`requirements.txt`**: Base development packages
- **`requirements-export.txt`**: AI/ML packages for ONNX export  
- **`requirements-server.txt`**: Web server and API dependencies
- **`requirements-tensorrt.txt`**: TensorRT optimization packages

## 🚀 Quick Start

### 1. Export Models to ONNX

```bash
# Export all models
python export/text_embedding/export_onnx.py --model_name sentence-transformers/all-MiniLM-L6-v2
python export/clip/export_image_encoder.py --model_name openai/clip-vit-base-patch32  
python export/clip/export_text_encoder.py --model_name openai/clip-vit-base-patch32
python export/vision/export_onnx.py --model_name resnet50 --feature_extraction
python export/reranker/export_onnx.py --model_name ms-marco-MiniLM-L-6-v2
```

### 2. Build TensorRT Engines

```bash
# Build optimized engines for target GPU
python tensorrt/build_engine.py --config tensorrt/configs/text_embedding.yaml
python tensorrt/build_engine.py --config tensorrt/configs/clip_image.yaml  
python tensorrt/build_engine.py --config tensorrt/configs/vision.yaml
python tensorrt/build_engine.py --config tensorrt/configs/reranker.yaml
```

### 3. Deploy with Docker

```bash
# Build and run the inference server
docker build -f docker/Dockerfile.server -t dgx-inference-server .
docker run --gpus all -p 8000:8000 -p 8001:8001 dgx-inference-server
```

### 4. Use the API

```python
import requests

# Text embedding
response = requests.post('http://localhost:8000/embed/text', 
                        json={'texts': ['Hello world', 'How are you?']})
embeddings = response.json()['embeddings']

# Image embedding  
with open('image.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
    
response = requests.post('http://localhost:8000/embed/image',
                        json={'images': [img_b64]})
image_embeddings = response.json()['embeddings']

# Document reranking
response = requests.post('http://localhost:8000/rerank', json={
    'query': 'machine learning algorithms',
    'documents': ['Neural networks are...', 'Linear regression is...'],
    'top_k': 5
})
ranked_docs = response.json()['ranked_documents']
```

## 📊 Performance Optimization

### TensorRT Engine Configuration

Models are optimized with:
- **FP16 precision** for 2x speedup
- **Dynamic batching** for throughput
- **Multi-GPU support** for scaling
- **Memory optimization** for large models

### Triton Inference Server Features

- **Concurrent model execution**
- **Request batching and queuing**
- **A/B testing and model versioning**
- **Prometheus metrics**
- **Health monitoring**

## 🧪 Benchmarking

### Latency Testing
```bash
cd benchmarks
python latency_test.py --server-url http://localhost:8000 --num-requests 1000
```

### Throughput Testing  
```bash
cd benchmarks
python throughput_test.py --server-url http://localhost:8000 --duration 60 --max-concurrent 50
```

## 🔧 Model Support

### Text Embedding Models
- Sentence Transformers (all-MiniLM, all-mpnet, etc.)
- BERT variants (DistilBERT, RoBERTa, etc.)
- Custom transformer models

### Vision Models
- **CLIP**: OpenAI CLIP variants
- **ResNet**: ResNet18/34/50/101/152
- **Vision Transformers**: ViT-Base/Large
- **EfficientNet**: B0-B7 variants

### Reranking Models
- Cross-encoder models
- MS MARCO fine-tuned models  
- Custom reranking architectures

## 🐳 Docker Deployment

### Multi-stage deployment:

1. **Export Stage**: `Dockerfile.export` - Model conversion to ONNX
2. **TensorRT Stage**: `Dockerfile.trt` - Engine optimization  
3. **Server Stage**: `Dockerfile.server` - Production deployment

### Production deployment:
```bash
# Complete pipeline
docker-compose up --build
```

## 📈 Monitoring & Observability

- **Health endpoints**: `/health`, `/models`
- **Prometheus metrics**: Request latency, throughput, errors
- **Structured logging**: JSON logs for centralized collection  
- **Performance profiling**: Built-in benchmarking tools

## 🔒 Security Features

- **Input validation**: Pydantic schemas
- **Rate limiting**: Per-client request throttling
- **CORS configuration**: Cross-origin request handling
- **Authentication**: JWT token support (optional)

## 🗂️ Configuration Management

Models configured via YAML:
```yaml
# tensorrt/configs/text_embedding.yaml
model_name: "text_embedding"
onnx_path: "../onnx_models/text_embedding/1/model.onnx"  
engine_path: "../tensorrt/engines/text_embedding/1/model.plan"

build_config:
  max_workspace_size: 4096  # MB
  fp16: true
  optimization_level: 5
  dynamic_shapes:
    input_ids:
      min: [1, 1]
      opt: [8, 512]  
      max: [64, 512]
```

## 📋 Requirements

### System Requirements
- **NVIDIA GPU**: RTX 30/40 series, A100, H100
- **CUDA**: ≥11.8
- **TensorRT**: ≥8.6
- **Docker**: ≥20.10 (for containerized deployment)

### Python Dependencies
- PyTorch ≥2.1.0
- Transformers ≥4.35.0  
- TensorRT ≥8.6.0
- Triton Client ≥2.40.0
- FastAPI ≥0.104.0

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/dgx-inference-server/issues)
- **Documentation**: [Wiki](https://github.com/your-org/dgx-inference-server/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/dgx-inference-server/discussions)