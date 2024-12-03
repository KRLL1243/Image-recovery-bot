import os
import telebot
import image_recovery

TOKEN = ''  # Insert your token here
bot = telebot.TeleBot(TOKEN)

is_recovering = False


@bot.message_handler(commands=['start'])
def main(message):
    if message.chat.type == 'private':
        bot.send_message(message.chat.id, f'Hi, {message.from_user.first_name}, I am a bot for recovering blurry images! Type /help to see a list of commands.')


@bot.message_handler(commands=['help'])
def main(message):
    if message.chat.type == 'private':
        bot.send_message(message.chat.id, '<b>Command list:</b>\n\n'
                                          '/recover - command for image recovery\n'
                                          'CAUTION: For a quality result, the image must be square!', parse_mode='html')


@bot.message_handler(commands=['recover'])
def recover(message):
    global is_recovering
    if message.chat.type == 'private':
        if not is_recovering:
            bot.send_message(message.chat.id, 'Send the image you want to recover.')
            is_recovering = True
        else:
            bot.send_message(message.chat.id, 'The image is already being processed.')


@bot.message_handler(content_types=['photo'])
def photo(message):
    global is_recovering
    if message.chat.type == 'private' and is_recovering:
        try:
            bot.send_message(message.chat.id, 'Processing...')

            photo_info = message.photo[-1]

            file_id = photo_info.file_id
            file_path = bot.get_file(file_id).file_path

            downloaded_file = bot.download_file(file_path)

            save_path = 'users_img'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            file_name = f'{file_id}.jpg'

            with open(os.path.join(save_path, file_name), 'wb') as new_file:
                new_file.write(downloaded_file)

            image_recovery.recover(img_path=f'users_img/{file_id}.jpg', out_path=f'users_img/{file_id}_recovered.jpg')

            bot.send_photo(message.chat.id, open(f"users_img/{file_id}_recovered.jpg", 'rb'))

            bot.delete_message(message.chat.id, message.message_id+1)
            bot.send_message(message.chat.id, 'Processing completed successfully!!!')
        except:
            bot.send_message(message.chat.id, 'Processing error!!!')
        finally:
            is_recovering = False
    else:
        bot.send_message(message.chat.id, 'Please use the /recover command if you want to restore the image.')


@bot.message_handler()
def other(message):
    if message.chat.type == 'private':
        bot.send_message(message.chat.id, 'Unknown command! Type /help to see a list of commands.')


bot.polling(none_stop=True)
