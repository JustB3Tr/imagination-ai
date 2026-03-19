# Colab setup for Imagination v1.1.2

## 1. GPU runtime
Runtime -> Change runtime type -> GPU -> Save

## 2. Install deps
```python
!pip install -q gradio transformers accelerate safetensors bitsandbytes requests beautifulsoup4 ddgs bcrypt diffusers
```

## 3. Mount Drive + expose repo
```python
from google.colab import drive
drive.mount("/content/drive")
!ln -s "/content/drive/MyDrive/imagination-v1.1.0" "/content/imagination-v1.1.0"
```

Or copy for faster I/O:
```python
!cp -r "/content/drive/MyDrive/imagination-v1.1.0" "/content/imagination-v1.1.0"
```

## 4. Launch
```python
%cd /content/imagination-v1.1.0/versions/v1.1.2
!python imagination_v1_1_2_colab_gradio.py
```

## Compute budget
~58h/month on L4 (100 pts at 1.71/hr). Shown in the hero bar.
