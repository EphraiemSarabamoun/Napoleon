from src.interfaces.ai_assistant import AIAssistant

if __name__ == "__main__":
    try:
        assistant = AIAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        from src.utils.config import logger
        logger.exception("Unexpected error")
    finally:
        if 'assistant' in locals():
            assistant.cleanup() 