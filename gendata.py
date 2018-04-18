import numpy as np
import string, os
from PIL import Image, ImageDraw, ImageFont

IMAGE_FOLDER = '/Images/'
FONT_FOLDER = os.getcwd()+'/Font Pack/'
fonts = os.listdir(FONT_FOLDER)

characters = list(string.ascii_letters[0:26]) #+list(string.digits)
print ('Characters: ',characters)
print ('Number of fonts used: ',len(fonts)-3)

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

xoff = (0,18)
yoff = (-5,6)
step = 3

for i,text in enumerate(characters):
    for idx,font_name in enumerate(fonts):
        font_loc = FONT_FOLDER+font_name  
        if font_name == '.DS_Store':
            continue      
        if not os.path.isfile(font_loc) or font_name.split('.')[1] != 'ttf':
            continue
        font = ImageFont.truetype(font_loc, 18)
        # offset = (7,5)
        max_size = (28,28)
        font_name = font_loc.split('.')[0].split('/')[-1]
        for offset_i in range(xoff[0],xoff[1],step):
            for offset_j in range(yoff[0],yoff[1],step):
                name = str(text)+str(count)+'.png'                    
                MakeImg(text, font, name, max_size, (offset_i,offset_j))
                count = count+1                
                Y.append(name +','+str(i)+','+text+',('+str(offset_i)+' '+str(offset_j)+'),'+font_name)

#Write CSV file
with open('Train.csv', 'w') as F:
    F.write('image,label,character,offset,font\n')
    F.write('\n'.join(Y))
