# -*- coding: utf-8 -*-

import os
import sys
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("telegram"))
sys.path.insert(0, root)
import telebot
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.dirname(cwd)))
sys.path.insert(0, cwd)
import config

USER_CHAT_ID = config.USER_CHAT_ID
BOT_TOKEN = config.BOT_TOKEN
tb = telebot.TeleBot(BOT_TOKEN)

class MyBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        # chạy lệnh khi nhận đc mess như trong commands
        @self.bot.message_handler(commands=['start', 'hello'])
        def send_welcome(message):
            self.send_welcome_message(message)

    # gửi thông báo kèm hình ảnh
    def send_notification(self, text, path_image = None, chat_id = USER_CHAT_ID):
        if path_image:
            with open(path_image, 'rb') as photo:
                self.bot.send_photo(chat_id, photo)
        self.bot.send_message(chat_id, text)

    # test mesage
    def send_welcome_message(self, message):
        self.bot.reply_to(message, "Hello, how are you doing?")
    
    # cái này để ghi nhận sự kiện hay sao ý, quên rùi
    def start_polling(self):
        self.bot.polling(none_stop=True)


# Chạy bot
if __name__ == "__main__":
    mybot = MyBot(token= BOT_TOKEN)
    mybot.start_polling()    