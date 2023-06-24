from io import BytesIO
import os
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from PIL import Image
import telebot

# Load token from .env file
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

bot = telebot.TeleBot(TOKEN, parse_mode=None)

@bot.message_handler(commands=["start"])
def send_help_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="This bot classifies Motif Batik on images. Send a motif batik photo")

@bot.message_handler(commands=["help"])
def send_help_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="This bot can classifies Motif Batik Kawung, Motif Batik Lasem, Motif Batik Parang, Motif Batik Mega Mendung, Motif Batik Sekar Jagad")
    
@bot.message_handler(content_types=["sticker"])
def send_sticker_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="Sorry, i can't identify sticker")
    
@bot.message_handler(content_types=["text"])
def send_emoticon_message(msg):
    bot.send_message(chat_id=msg.chat.id, text="Sorry, i can't response your text")


# Load the model
model_path = 'model/model.pb'
labels_path = 'model/labels.txt'

            
# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TensorFlow model
graph = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(model_path, 'rb') as model_file:
    graph.ParseFromString(model_file.read())

session = tf.compat.v1.Session(graph=tf.Graph())

# Fungsi untuk melakukan klasifikasi gambar
def classify_image(image):
    # Mengubah gambar ke dalam format TensorFlow
    img = Image.open(BytesIO(image))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape((1, 224, 224, 3))

    # Melakukan prediksi
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph, name='')
        input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
        output_tensor = sess.graph.get_tensor_by_name('loss:0')
        predictions = sess.run(output_tensor, {input_tensor: img})

    predicted_label = labels[np.argmax(predictions)]

    return predicted_label

# Menangani pesan gambar yang diterima
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
    bot.reply_to(msg, f"The image classified as a Motif Batik : {predicted_label}")

bot.polling()