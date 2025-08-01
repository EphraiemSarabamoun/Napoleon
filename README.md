# Napoleon AI Chat

A local AI assistant with memory, webcam capture, and customizable personalities.
All code is currently in Chat_Mode folder. Possible agent folder will be added in the future.

## Quick Start

1. **Install Ollama** and pull the model:
   ```bash
   ollama pull deepseek-r1:32b
   ollama pull llava:13b
   ```

2. **Install pipenv** (if not already installed):
   ```bash
   pip install pipenv
   ```

3. **Install dependencies** using the Pipfile:
   ```bash
   cd Napoleon/Chat_Mode
   pipenv install
   ```

4. **Run the app**:
   ```bash
   pipenv run python main-app.py
   ```

5. **Open your browser** to `http://localhost:5000`

## Features

- **Chat with AI** using local Ollama models
- **Memory system** remembers past conversations
- **Webcam capture** - take photos and analyze them
- **Custom personalities** - create different AI personas
- **Clean UI** with modern design

## Usage

- **Type messages** in the chat box
- **Capture webcam** with the camera button
- **Manage personalities** in the right panel
- **Toggle thinking** to hide/show AI reasoning

## File Structure

- `main-app.py` - Main Flask application
- `image_handling.py` - Webcam and image analysis
- `static/index.html` - Web interface
- `memories/` - Saved conversation memory
- `output_frames/` - Captured webcam images
- `personalities.json` - Custom AI personalities

## Requirements

- Python 3.13 for the pipfile 
- Ollama installed locally
- Webcam (optional)
- Internet connection (for initial setup)
