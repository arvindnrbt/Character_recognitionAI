import numpy as np
import string, os
from PIL import Image, ImageDraw, ImageFont

IMAGE_FOLDER = '/Images/'
FONT_FOLDER = os.getcwd()+'/Font Pack/'
fonts = os.listdir(FONT_FOLDER)

img_row = 50
img_col = 50

characters = list(string.ascii_letters[0:26]) #+list(string.digits)
print ('Characters: ',characters)
print ('Number of fonts used: ',len(fonts)-3)

Y = []
count = 0

def MakeImg(text, font, name, size, offset):
    img = Image.new('RGB', (size[0]-2,size[1]-2), "white")
    draw = ImageDraw.Draw(img)
    color = (0,0,0)
    draw.text(offset, text, color, font = font)
    directory = os.getcwd()
    if not os.path.exists(directory+IMAGE_FOLDER):
        os.makedirs(directory+IMAGE_FOLDER)
    img = img.resize((img_row, img_col))
    img.save(directory+name)

xoff = (0,6)
yoff = (-5,4)
step = 3

for i,text in enumerate(characters):
    for idx,font_name in enumerate(fonts):
        font_loc = FONT_FOLDER+font_name  
        if font_name == '.DS_Store':
            continue      
        if not os.path.isfile(font_loc) or font_name.split('.')[1] != 'ttf':
            continue
        font = ImageFont.truetype(font_loc, 53)
        # offset = (7,5)
        size = font.getsize(text)
        font_name = font_loc.split('.')[0].split('/')[-1]
        for offset_i in range(xoff[0],xoff[1],step):
            for offset_j in range(yoff[0],yoff[1],step):
                name = IMAGE_FOLDER+str(text)+str(count)+'.png'                    
                MakeImg(text, font, name, size, (offset_i,offset_j))
                count = count+1                
                Y.append('.'+name +','+str(i)+','+text+',('+str(offset_i)+' '+str(offset_j)+'),'+font_name)

#Write CSV file
with open('Train.csv', 'w') as F:
    F.write('image,label,character,offset,font\n')
    F.write('\n'.join(Y))
