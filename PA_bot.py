from aiogram import Bot, Dispatcher, executor, types
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
 
API_TOKEN = '5930734456:AAHasa5rrpH5HoWHSNZRYG7yMHyjW8Jpw8I'
 
MODEL_FILENAME = 'my_model.h5'
ORDER_FILENAME = 'order'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
   await message.reply("Привет, пришли мне фото цветка, а я опредлю его вид")
 
@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
   await message.photo[-1].download(destination_file='img/ggg.jpg')
   ###
   class ModelWrapper:
    __instance = None
    def __init__(self):
        if ModelWrapper.__instance is not None:
            raise Exception("This class is a singleton!")
        ModelWrapper.__instance = self
        self.model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(ORDER_FILENAME) as f:
            self.order_tags = f.readline().split(',')[:5]
            
    @staticmethod
    def get_shared():
        if ModelWrapper.__instance is None:
            ModelWrapper()
        return ModelWrapper.__instance

    def predict(self, filename):
        size = 64,64
        img = cv2.imread(filename)
        im = np.array([cv2.resize(img, size)])
        im = im.astype('float32') / 255.0
        poss = self.model.predict(im)
        ind = np.argmax(poss)
        return self.order_tags[ind]

   if __name__ == "__main__":
      model = ModelWrapper.get_shared()
      await message.reply(model.predict("img/ggg.jpg"))
   ###

if __name__ == '__main__':
   executor.start_polling(dp, skip_updates=True)