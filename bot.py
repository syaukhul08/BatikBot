from io import BytesIO
import os
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from PIL import Image
import telebot
from predict import _predict_image, _initialize


# Load token from .env file
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

bot = telebot.TeleBot(TOKEN, parse_mode=None)

@bot.message_handler(commands=["start"])
def send_help_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="This bot classifies Motif Batik on images. Send a motif batik photo")

@bot.message_handler(commands=["help"])
def send_help_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="This bot can classifies\n - Motif Batik Kawung,\n - Motif Batik Lasem,\n - Motif Batik Parang,\n - Motif Batik Mega Mendung,\n - Motif Batik Sekar Jagad\n\nCan classify motifs with an accuracy level of >= 90%")
    
@bot.message_handler(content_types=["sticker"])
def send_sticker_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="Sorry, i can't identify sticker")
    
@bot.message_handler(content_types=["text"])
def send_emoticon_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="Sorry, i can't response your text")
    

def classify_image(image):
    
    _initialize()
    
    return _predict_image(image) 


@bot.message_handler(content_types=['photo'])
def handle_image(msg):
    # Mendapatkan file gambar dari pesan
    file_id = msg.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    # Mendownload gambar
    downloaded_file = bot.download_file(file_path)

    # Melakukan klasifikasi gambar
    predicted_label = classify_image(downloaded_file)

    # Mengirim hasil klasifikasi ke pengguna
    bot.reply_to(msg, f"The image classified as a Motif Batik : \n{predicted_label}")

bot.polling()