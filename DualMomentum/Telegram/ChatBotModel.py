
# coding: utf-8
import telegram
#from telegram.ext import Updater, CommandHandler, MessageHandler
from telegram.ext import *

class TelegramBot:
    def __init__(self, name, token):
        self.core = telegram.Bot(token)
        self.updater = Updater(token)
        self.id = 571675744
        self.name = name

    def sendMessage(self, text):
        self.core.sendMessage(chat_id = self.id, parse_mode=None,text=text)

    def sendPhoto(self, photo):
        self.core.sendPhoto(chat_id = self.id, photo=photo)

    def stop(self):
        self.updater.start_polling()
        self.updater.dispatcher.stop()
        self.updater.job_queue.stop()
        self.updater.stop()

class BotDual (TelegramBot):
    def __init__(self):
        self.token = "629390046:AAET-uyD4ph8eyDCBAo1I3WjxwFt96OYFJs"
        TelegramBot.__init__(self, 'dual', self.token)
        self.updater.stop()

    def add_handler(self, cmd, func):
        self.updater.dispatcher.add_handler(CommandHandler(cmd, func))

    def start(self):
        self.sendMessage('Dualmomentum Stock & Bond & Cash')
        self.updater.start_polling()
        self.updater.idle()
