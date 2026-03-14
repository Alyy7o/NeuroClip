# Task: Setup Custom Computer Vision Training Pipeline

## Context
Implementation of a custom "Two-Tower" Vision-Language model training pipeline (CLIP style) using PyTorch, designed to be trained on datasets like MSR-VTT.

## Todo List
- [ ] Create directory structure `backend/custom_model_training`
- [ ] Implement `model.py` (Dual Encoder Architecture: ResNet50 + DistilBERT)
- [ ] Implement `dataset.py` (MSR-VTT / CSV Data Loader)
- [ ] Implement `train.py` (Training Loop with Contrastive Loss)
- [ ] Create `requirements_training.txt` for reproducibility
- [ ] Add `README.md` with usage instructions and folder documentation
- [ ] Copy architectural guides/docs into `backend/custom_model_training/docs`
