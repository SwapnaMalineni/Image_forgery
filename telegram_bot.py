from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import requests
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
from PIL import Image

# ✅ Set correct bot token
TELEGRAM_BOT_TOKEN = "8186463364:AAFLIrtKvCL24RuN-_1SUDv1XW6UW1cwgKs"

# ✅ Set correct Flask API URL (Replace with your FULL Ngrok URL)
FLASK_API_URL = "https://6dd0-2405-201-c05e-a87b-645a-3ef0-3489-2053.ngrok-free.app/api/detect-forgery"

# ✅ Manually set the timezone to UTC to avoid system conflicts
os.environ['TZ'] = 'UTC'

# ✅ Ensure APScheduler uses pytz timezone
scheduler = AsyncIOScheduler(timezone=pytz.UTC)

async def start(update: Update, context: CallbackContext):
    """Sends a welcome message when the bot is started."""
    await update.message.reply_text(
        "👋 Welcome to the Image Forgery Detection Bot! 🖼️\n\n"
        "📌 Send me an image, and I'll analyze it for authenticity."
    )
    
def format_response(result):
    """Formats the API response to send back to the user."""
    return (
        f"🔍 *Analysis Results:*\n"
        f"- *Status:* {result['result']}\n"
        f"- *Confidence:* {result['confidence']:.2f}%\n"
        f"- *Metadata:*\n"
        f"  - Size: {result['metadata']['size']}\n"
        f"  - Format: {result['metadata']['format']}\n"
        f"  - Software: {result['metadata']['software']}"
    )


async def handle_image(update: Update, context: CallbackContext):
    """Handles image uploads as files to avoid compression."""
    if update.message.document:  # If user sends as file
        file = await update.message.document.get_file()
    else:  # If user sends as photo (compressed)
        file = await update.message.photo[-1].get_file()
    
    file_path = f"uploads/{file.file_id}.jpg"
    await file.download_to_drive(file_path)

    # ✅ Print image details for debugging
    img = Image.open(file_path)
    print(f"✅ Telegram Image: {file_path}")
    print(f"✅ Telegram Image: {file_path}")
    print(f"   - Size: {img.size}")
    print(f"   - Format: {img.format}")
    print(f"   - Mode: {img.mode}")

    try:
        print(f"✅ Sending image to Flask API: {FLASK_API_URL}")
        with open(file_path, 'rb') as image_file:
            response = requests.post(FLASK_API_URL, files={'file': image_file})

        print(f"✅ Full API Response: {response.status_code} - {response.text}")

        if response.status_code == 200:
            result = response.json()
            reply_text = format_response(result)
        else:
            reply_text = f"❌ Error analyzing the image.\nServer Response: {response.text}"

        await update.message.reply_text(reply_text, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def main():
    """Initializes and starts the Telegram bot."""
    
    global scheduler 
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.job_queue.scheduler = scheduler  # ✅ Set scheduler timezone

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == "__main__":
    main()
