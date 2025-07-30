# Napoleon AI Chat Mode - Complete Documentation

A comprehensive local AI assistant with persistent memory, web interface, and advanced image/video processing capabilities.

## Overview

Napoleon AI Chat Mode provides a fully local AI assistant that combines conversational AI with computer vision capabilities. The system uses DeepSeek R1 32B via Ollama for intelligent responses and includes specialized tools for document processing, video analysis, and image text extraction.

## Architecture

### Core Components
- **Main Application** (`main-app.py`): Flask-based web server with REST API
- **AI Model**: DeepSeek R1 32B via Ollama
- **Memory System**: Persistent neural memory with PyTorch-based embeddings
- **Web Interface**: Responsive Bootstrap-based UI
- **Computer Vision**: OpenCV + EasyOCR for text extraction

## Features

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

## Installation & Setup

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

## Quick Start Guide

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

### Configuration
- **Model**: deepseek-r1:32b
- **Memory Size**: 10 slots
- **Embedding**: 256 dimensions
- **Port**: 5000 (configurable)
