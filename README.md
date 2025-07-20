<<<<<<< HEAD
# napoleon
AI organizer for PC, laptop, and phone.
=======
# Napoleon
An all-in-one local AI

## Web Interface for Napoleon AI

This project provides a user-friendly web interface for interacting with the Napoleon AI model. The interface allows you to chat with the AI and see the memory entries that influence its responses.

### Features

- Clean, modern UI for chatting with Napoleon AI
- Real-time display of memory entries related to your queries
- Responsive design that works on desktop and mobile

### Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the Flask server:
   ```
   python server.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

4. Start chatting with Napoleon AI!

### How It Works

The web interface connects to a Flask backend that interfaces with the Napoleon AI model. When you send a message:

1. The message is stored in the AI's memory
2. The AI retrieves related memories to provide context
3. The AI generates a response based on your message and its memory
4. The response and related memory entries are displayed in the interface

### Requirements

- Python 3.8 or higher
- Ollama installed with the DeepSeek model available
- Web browser with JavaScript enabled
>>>>>>> 4ac9c1e (Commiting all local files)
