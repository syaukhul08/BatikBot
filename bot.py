from io import BytesIO
import os
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from PIL import Image
import telebot
from predict import _convert_to_nparray, _crop_center, _extract_and_resize_to_224_square, _initialize, _log_msg, _resize_down_to_1600_max_dim, _update_orientation
from predict import _predict_image
from predict import *


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


scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
filename = os.path.join(scriptdir, 'model/model.pb')
labels_filename = os.path.join(scriptdir, 'model/labels.txt')


output_layer = 'loss:0'
input_node = 'Placeholder:0'

graph_def = tf.compat.v1.GraphDef()
labels = []
network_input_size = 0


def _predict_image(image):
    try:
        if image.mode != "RGB":
            _log_msg("Converting to RGB")
            image.convert("RGB")

        w,h = image.size
        _log_msg("Image size: " + str(w) + "x" + str(h))
        
        # Update orientation based on EXIF tags
        image = _update_orientation(image)

        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimention is 1600
        image = _resize_down_to_1600_max_dim(image)

        # Convert image to numpy array
        image = _convert_to_nparray(image)
        
        # Crop the center square and resize that square down to 256x256
        resized_image = _extract_and_resize_to_224_square(image)

        # Crop the center for the specified network_input_Size
        cropped_image = _crop_center(resized_image, network_input_size, network_input_size)

        _initialize()

        tf.compat.v1.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

        with tf.compat.v1.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [cropped_image] })
            
            result = []
            highest_prediction = None
            for p, label in zip(predictions, labels):
                truncated_probablity = np.float64(round(p,8))
                if truncated_probablity > 1e-8:
                    prediction = {
                        'tagName': label,
                        'probability': truncated_probablity }
                    result.append(prediction)
                    if not highest_prediction or prediction['probability'] > highest_prediction['probability']:
                        highest_prediction = prediction

            response = {
                'created': datetime.utcnow().isoformat(),
                'predictedTagName': highest_prediction['tagName'],
                'prediction': result 
            }

            _log_msg("Results: " + str(response))
            return response
            
    except Exception as e:
        _log_msg(str(e))
        return 'Error: Could not preprocess image for prediction. ' + str(e)

@bot.message_handler(content_types=['photo'])
def handle_image(msg):
    # Mendapatkan file gambar dari pesan
    file_id = msg.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    # Mendownload gambar
    downloaded_file = bot.download_file(file_path)

    # Melakukan klasifikasi gambar
    predicted_label = _predict_image(downloaded_file)

    # Mengirim hasil klasifikasi ke pengguna
    bot.reply_to(msg, f"Predicted label: {predicted_label}")

bot.polling()