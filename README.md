# Napoleon AI Chat Mode - Complete Documentation

A comprehensive local AI assistant with persistent memory, web interface, and advanced image/video processing capabilities.

## 🚀 Overview

Napoleon AI Chat Mode provides a fully local AI assistant that combines conversational AI with computer vision capabilities. The system uses DeepSeek R1 32B via Ollama for intelligent responses and includes specialized tools for document processing, video analysis, and image text extraction.

## 🏗️ Architecture

### Core Components
- **Main Application** (`main-app.py`): Flask-based web server with REST API
- **AI Model**: DeepSeek R1 32B via Ollama
- **Memory System**: Persistent neural memory with PyTorch-based embeddings
- **Web Interface**: Responsive Bootstrap-based UI
- **Computer Vision**: OpenCV + EasyOCR for text extraction

### Project Structure
```
napoleon/Chat_Mode/
├── main-app.py              # Main Flask web application
├── image_handling.py        # Webcam capture and OCR analysis
├── live_video_text_extraction.py  # Real-time video text detection
├── video_text_extraction.py # Pre-recorded video text processing
├── image_text_extraction.py # Static image text extraction
├── wallpaper.py            # Custom physics-themed wallpaper generator
├── static/                 # Web assets (index.html)
├── memories/               # Persistent memory storage
├── output_frames/          # Captured images and processed frames
└── README.md              # This documentation
```

## 🎯 Features

### Core Chat Functionality
- **Persistent Memory**: Remembers conversations across sessions
- **Context Awareness**: Retrieves relevant memories for responses
- **Name Recognition**: Automatically learns and remembers user names
- **Real-time Interface**: Modern web-based chat UI

### Image & Video Processing
- **Webcam Text Extraction**: Real-time OCR from live camera feed
- **Image Text Recognition**: Extract text from static images
- **Video Text Processing**: Batch processing of video files
- **Live Video Analysis**: Continuous text detection from webcam
- **Frame Storage**: Organized saving of captured images

### Specialized Tools
- **Document Processing**: OCR for images and documents
- **Video Analysis**: Extract text from video frames
- **Wallpaper Management**: Generate custom physics-themed wallpapers

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install flask opencv-python easyocr torch numpy
```

### Setup Ollama
Ensure Ollama is installed and running with the DeepSeek model:
```bash
ollama pull deepseek-r1:32b
```

## 🚀 Quick Start

### 1. Start the Web Chat Interface
```bash
python main-app.py
```
Access at: http://localhost:5000

### 2. Webcam Text Analysis
```bash
python image_handling.py
```
This will:
- Open your webcam
- Capture an image after 3 seconds
- Save it to output_frames/ with timestamp
- Analyze and display any detected text

### 3. Live Video Text Detection
```bash
python live_video_text_extraction.py
```
Opens a real-time window showing text detection from your webcam.

### 4. Video File Processing
```bash
# Place a video.mp4 file in the directory
python video_text_extraction.py
```

### 5. Image Text Extraction
```bash
python image_text_extraction.py
```

## 📊 Usage Examples

### Webcam Capture & Analysis
```python
# The image_handling.py script provides:
# - Automatic webcam access
# - Timestamp-based file naming
# - OCR text extraction
# - Organized storage in output_frames/
```

### Memory Management
The system automatically saves conversations to `memories/dnc_memory.pt` and loads them on startup.

## 🔧 Configuration

### Camera Settings
- Default camera: 0 (built-in webcam)
- Resolution: 640x480 for optimal performance
- Frame processing: Every 30th frame for videos

### OCR Settings
- Language: English ('en')
- GPU: Disabled by default (set to True if CUDA available)
- Confidence threshold: Variable based on text clarity

## 📁 File Organization

### Output Structure
```
output_frames/
## 🎯 Current Chat_Mode Features

### Core AI Assistant
- **Dual Interface**: Both terminal (`main.py`) and web (`main-app.py`) interfaces
- **Persistent Memory**: Neural memory system with PyTorch embeddings
- **Name Recognition**: Automatically learns and remembers user names
- **Context Awareness**: Retrieves relevant memories for responses

### Image & Video Processing
- **Webcam Text Extraction**: Real-time OCR from live camera (`image_handling.py`)
- **Live Video Analysis**: Continuous text detection from webcam
- **Video File Processing**: Batch processing of pre-recorded videos
- **Static Image OCR**: Extract text from saved images
- **Organized Storage**: Timestamped captures in output_frames/

## 🏗️ Current Project Structure

```
napoleon/
├── README.md                    # This documentation
├── Chat_Mode/                   # AI Chat & Processing Module
│   ├── main.py                  # Terminal-based AI assistant
│   ├── main-app.py              # Flask web server with REST API
│   ├── image_handling.py        # Webcam capture + OCR analysis
│   ├── static/                  # Web assets
│   │   └── index.html          # Chat web interface
│   ├── memories/               # Persistent memory storage
│   │   └── dnc_memory.pt       # Neural memory file
│   ├── output_frames/          # Captured images and processed frames
│   └── .gitignore              # Git ignore file
└── .git/                       # Git repository
```

## 🚀 Quick Start Guide

### Terminal Interface
```bash
cd Chat_Mode
python main.py
```
- **Interactive**: Direct terminal conversation
- **Name Learning**: Automatically learns your name
- **Memory**: Remembers conversations across sessions

### Web Interface
```bash
cd Chat_Mode
python main-app.py
```
- **URL**: http://localhost:5000
- **REST API**: POST /api/chat endpoint
- **Real-time**: Modern web UI with memory display

### Webcam Image Processing
```bash
cd Chat_Mode
python image_handling.py
```
- **Capture**: Takes photo from webcam after 3 seconds
- **Analysis**: Runs OCR to extract text
- **Storage**: Saves to output_frames/ with timestamp
- **Results**: Displays detected text and confidence scores

## 🔧 Technical Details

### Dependencies
- **AI Engine**: DeepSeek R1 32B via Ollama
- **Memory**: PyTorch-based neural embeddings
- **OCR**: EasyOCR for text extraction
- **Web**: Flask REST API with Bootstrap frontend
- **Camera**: OpenCV for webcam access

### Configuration
- **Model**: deepseek-r1:32b
- **Memory Size**: 10 slots
- **Embedding**: 256 dimensions
- **Port**: 5000 (configurable)