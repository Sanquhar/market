from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hello! Welcome to my Telegram bot.')


class bot_telegram:

    def __init__(self) -> None:
        token = "7590818864:AAEUkX5pJrWSAO2SQs7KUMScFBZR16tGQDA"
        app = ApplicationBuilder().token(token).build()
        app.add_handler(CommandHandler("start", start))

        print("start")
        app.run_polling()

        pass

   