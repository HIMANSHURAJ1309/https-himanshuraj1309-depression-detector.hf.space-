---
title: Depression Detector
emoji: 👀
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
---

# Depression Detector

> **Multi-level Depression Detection From Social Media Posts**
> 
> An explainable AI system using MENTALRoBERTa-BiLSTM for detecting depression levels from social media content.

🔗 **Live Demo:** [himanshuraj1309-depression-detector.hf.space](https://himanshuraj1309-depression-detector.hf.space/)

## Features

- 🤖 Multi-level depression detection using advanced NLP
- 📊 MENTALRoBERTa-BiLSTM architecture for improved accuracy
- 🔍 Explainable predictions with interpretability
- 🌐 Analyzes social media posts
- 💻 Gradio-based web interface for easy interaction

## How It Works

This application uses a specialized BERT model (MENTALRoBERTa) combined with BiLSTM layers to detect depression severity levels from text. The model provides not just predictions but also explanations for its decisions.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Usage

- Enter or paste social media text
- The model will analyze the content and provide:
  - Depression level classification
  - Confidence scores
  - Feature importance/explanations

## Model Details

- **Architecture:** MENTALRoBERTa-BiLSTM
- **Task:** Multi-level depression classification
- **Input:** Social media text
- **Output:** Depression level with explainability features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created by [HIMANSHURAJ1309](https://github.com/HIMANSHURAJ1309)

---

**Note:** This tool is for informational purposes. Please consult mental health professionals for actual mental health concerns.
