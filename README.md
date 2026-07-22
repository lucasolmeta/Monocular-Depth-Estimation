# Monocular Depth Estimation

Estimate depth from a single RGB image and reconstruct an interactive 3D point cloud.

This notebook uses Intel’s [DPT BEiT Large 512](https://huggingface.co/Intel/dpt-beit-large-512) model via Hugging Face `transformers`, with dynamic INT8 quantization of `nn.Linear` layers for lighter CPU inference. Predicted depth is back-projected into a colored Open3D point cloud and viewed with Plotly.

## Setup

```bash
pip install -r requirements.txt
```

Open `notebook.ipynb` and run the cells in order. The first run downloads the pretrained weights and may take a few minutes.

## Pipeline

1. Load and resize an image from `images/`
2. Run DPT depth inference
3. Invert / normalize the depth map and back-project pixels with a simple pinhole camera
4. Clean outliers and display the colored point cloud
