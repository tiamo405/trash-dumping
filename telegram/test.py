import os, sys
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("telegram"))
sys.path.insert(0, root)
from telegram.bot import MyBot

import config

BOT_TOKEN = config.BOT_TOKEN

myBot = MyBot(token=BOT_TOKEN)
def main():
    for i in range(5):
        myBot.send_notification("Hello, how are you doing?")
    myBot.send_notification("Hello, how are you doing?")
    # myBot.start_polling()

if __name__ == "__main__":
    main()