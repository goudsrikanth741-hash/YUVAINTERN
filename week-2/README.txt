Week 2 Task - AI Model Architecture Design
Automated Pneumonia Detection from Chest X-Rays (ResNet50 Transfer Learning CNN)

Contents:
1. AI_Model_Architecture_Design_Week2.docx
   - Full design report: research problem, architecture, diagrams,
     algorithm/layer explanations, design rationale, benefits/challenges,
     scalability and future modifications, references.

2. pneumonia_cnn.py
   - Complete, runnable PyTorch implementation of the model described
     in the report: data pipeline, ResNet50-based model, staged
     fine-tuning training loop, and evaluation (accuracy, precision,
     recall, F1, AUC, confusion matrix).
   - Usage: python pneumonia_cnn.py --data_dir ./data --epochs 15 --batch_size 32
   - Expected data layout: data/train|val|test/NORMAL|PNEUMONIA/*.jpg

3. figure1_architecture_diagram.png / figure2_data_pipeline_diagram.png
   - Standalone copies of the diagrams embedded in the report.
