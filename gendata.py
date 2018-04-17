import numpy as np
import string, os
from PIL import Image, ImageDraw, ImageFont

IMAGE_FOLDER = '/Images/'
FONT_FOLDER = './Font Pack/'
fonts = os.listdir(FONT_FOLDER)

characters = list(string.ascii_letters[0:26]) #+list(string.digits)
print ('Characters: ',characters)
print ('Number of fonts used: ',len(fonts)-1)

Y = []
count = 0

def MakeImg(text, font, name, max_size, offset):
    img = Image.new('RGB', max_size, "white")
    draw = ImageDraw.Draw(img)
    draw.text(offset, text, (0, 0, 0), font = font)
    directory = os.getcwd() + IMAGE_FOLDER
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.save(directory+name)

for i,text in enumerate(characters):
    for font_name in fonts:
        if font_name == '.DS_Store':
            continue
        font_loc = FONT_FOLDER+font_name
        font = ImageFont.truetype(font_loc, 16)
        offset = (7,5)
        max_size = (28,28)
        font_name = font_loc.split('.')[0].split('/')[-1]
        name = str(text)+str(count)+'.png'
        count = count+1        
        MakeImg(text, font, name, max_size, offset)
        Y.append(name +','+str(i)+','+text)

#Write CSV file
with open('Train.csv', 'w') as F:
    F.write('image,label,character\n')
    F.write('\n'.join(Y))
