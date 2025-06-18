import logging
import os
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from model_utils import ModelManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CONFIG = {
    "faiss_index_path": "faiss_data/games_data_index.index",
    "data_dir": "faiss_data/games_data",
    "review_model_name": "llama-3.2-3B-ign_rev_sft/final_vers"
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Help!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_manager = context.bot_data['model_manager']
    model_manager.cleanup()

    game_ids = model_manager.search_games(update.message.text)
    games_data = [model_manager.load_game_data(gid) for gid in game_ids]
    top_games = model_manager.rerank_games(update.message.text, games_data)

    for game in top_games:
        review = model_manager.generate_review(game)
        response = (
            f"{game['name']}\n"
            f"Price: {game['price']}\n"
            f"Description: {game['short_description']}\n"
            f"Review: {review}\n"
        )
        await update.message.reply_text(response)
        model_manager.cleanup()


def main():
    application = Application.builder().token(os.getenv("TG_STEAM_BOT_TOKEN")).build()

    model_manager = ModelManager(CONFIG).load_models()
    application.bot_data['model_manager'] = model_manager

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()