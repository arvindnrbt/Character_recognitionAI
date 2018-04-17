import numpy as np
import string, os
from PIL import Image, ImageDraw, ImageFont

def MakeImg(text, font, name, max_size, offset):
    img = Image.new('RGB', max_size, "white")
    draw = ImageDraw.Draw(img)
    draw.text(offset, text, (0, 0, 0), font = font)
    directory = os.getcwd() + '/Images/'
    img.save(directory+name)

fonts = ["/Library/Fonts/Arial.ttf", "/Library/Fonts/PlantagenetCherokee.ttf", "/Library/Fonts/Chalkduster.ttf",
"/Library/Fonts/Courier New Bold Italic.ttf", "/Library/Fonts/DIN Condensed Bold.ttf", '/Library/Fonts/Courier New.ttf', '/Library/Fonts/Arial Bold.ttf', '/Library/Fonts/Herculanum.ttf', '/Library/Fonts/Times New Roman Bold Italic.ttf',
'/Library/Fonts/Arial Italic.ttf', '/Library/Fonts/Times New Roman Bold.ttf', '/Library/Fonts/Times New Roman Italic.ttf', '/Library/Fonts/Times New Roman.ttf', '/Library/Fonts/Georgia Bold Italic.ttf', '/Library/Fonts/Georgia Bold.ttf',
'/Library/Fonts/Georgia.ttf', '/Library/Fonts/Arial Black.ttf', '/Library/Fonts/Arial Bold Italic.ttf', '/Library/Fonts/Arial Narrow.ttf', '/Library/Fonts/Arial Narrow Bold Italic.ttf', '/Library/Fonts/Arial Narrow Bold.ttf', '/Library/Fonts/Arial Narrow Italic.ttf',
'/Library/Fonts/Arial Unicode.ttf', '/Library/Fonts/Courier New Italic.ttf', '/Library/Fonts/Courier New Bold.ttf', '/Library/Fonts/Tahoma.ttf', '/Library/Fonts/AppleGothic.ttf', '/Library/Fonts/Arial Narrow Bold.ttf', '/Library/Fonts/Comic Sans MS Bold.ttf', '/Library/Fonts/Comic Sans MS.ttf',
'/Library/Fonts/DIN Alternate Bold.ttf', '/Library/Fonts/Microsoft Sans Serif.ttf', '/Library/Fonts/Verdana Bold Italic.ttf', '/Library/Fonts/Verdana Bold.ttf', '/Library/Fonts/Verdana Italic.ttf', '/Library/Fonts/Verdana.ttf', '/Library/Fonts/Brush Script.ttf', '/Library/Fonts/Tahoma Bold.ttf']


fonts = fonts + ['./Font Pack/12tonsushi.ttf','./Font Pack/Ozonelayer.ttf','./Font Pack/arialbd.ttf',
'./Font Pack/8bitlim.ttf','./Font Pack/PRIMER.ttf','./Font Pack/autobahn.ttf',
'./Font Pack/Aaargh.ttf','./Font Pack/PRIMERB.ttf','./Font Pack/basictitlefont.ttf',
'./Font Pack/Action Man Bold.ttf','./Font Pack/Pacifico.ttf','./Font Pack/belizarius.ttf',
'./Font Pack/Action_Force.ttf','./Font Pack/Permanent.ttf','./Font Pack/big_noodle_titling.ttf',
'./Font Pack/AdineKirnberg-Script.ttf','./Font Pack/Phantomime.ttf','./Font Pack/big_noodle_titling_oblique.ttf',
'./Font Pack/Adventure.ttf','./Font Pack/Qiroff.ttf','./Font Pack/black.ttf',
'./Font Pack/Advokat Modern.ttf','./Font Pack/REVOLUTION.ttf','./Font Pack/bluebold.ttf',
'./Font Pack/Aerovias Brasil NF.ttf','./Font Pack/Righteous-Regular.ttf','./Font Pack/bluecond.ttf',
'./Font Pack/Aileenation.ttf','./Font Pack/SF Arch Rival Bold.ttf','./Font Pack/bluehigh.ttf',
'./Font Pack/AldotheApache.ttf','./Font Pack/SF Atarian System.ttf','./Font Pack/bobcaygr.ttf',
'./Font Pack/Alice and the Wicked Monster.ttf','./Font Pack/SF Buttacup Lettering Bold Oblique.ttf','./Font Pack/bold.ttf',
'./Font Pack/Aliquam.ttf','./Font Pack/SF Buttacup Lettering Bold.ttf','./Font Pack/boston.ttf',
'./Font Pack/AliquamREG.ttf','./Font Pack/SF Buttacup Lettering.ttf','./Font Pack/bottix.ttf',
'./Font Pack/AliquamRit.ttf','./Font Pack/SF Cartoonist Hand SC Bold Italic.ttf','./Font Pack/bradybun.ttf',
'./Font Pack/Aliquamulti.ttf','./Font Pack/SF Cartoonist Hand.ttf','./Font Pack/bronic.ttf',
'./Font Pack/AllCaps.ttf','./Font Pack/SF Collegiate Solid.ttf','./Font Pack/budeasym.ttf',
'./Font Pack/Alpha Thin.ttf','./Font Pack/SF Espresso Shack Bold Italic.ttf','./Font Pack/cartoonist_kooky.ttf',
'./Font Pack/AmazOOSTROVFine.ttf','./Font Pack/SF Espresso Shack Bold.ttf','./Font Pack/charbb_reg.ttf',
'./Font Pack/American Captain.ttf','./Font Pack/SF Foxboro Script Bold.ttf','./Font Pack/chinese rocks rg.ttf',
'./Font Pack/AnkeHand.ttf','./Font Pack/SF Foxboro Script Extended Bold.ttf','./Font Pack/coolvetica rg.ttf',
'./Font Pack/Artbrush.ttf','./Font Pack/SF Foxboro Script.ttf','./Font Pack/corbel.ttf',
'./Font Pack/Atlantic_Cruise-Demo.ttf','./Font Pack/SF Grunge Sans Bold Italic.ttf','./Font Pack/corbelb.ttf',
'./Font Pack/BNMachine.ttf','./Font Pack/SF Grunge Sans Bold.ttf','./Font Pack/corbeli.ttf',
'./Font Pack/BabelSans-Bold.ttf','./Font Pack/SF Grunge Sans Italic.ttf','./Font Pack/corbelz.ttf',
'./Font Pack/BabelSans-BoldOblique.ttf','./Font Pack/SF Grunge Sans SC Italic.ttf','./Font Pack/criticized.ttf',
'./Font Pack/BabelSans-Oblique.ttf','./Font Pack/SF Grunge Sans SC.ttf','./Font Pack/deathrattlebb_reg.ttf',
'./Font Pack/BabelSans.ttf','./Font Pack/SF Hollywood Hills Bold Italic.ttf','./Font Pack/distortion_of_the_brain_and_mind.ttf',
'./Font Pack/Backpack_PersonalUse.ttf','./Font Pack/SF Hollywood Hills Bold.ttf','./Font Pack/djcourag.ttf',
'./Font Pack/BankGothic-Regular DB.ttf','./Font Pack/SF Iron Gothic Bold Oblique.ttf','./Font Pack/edo.ttf',
'./Font Pack/BankGothic-Regular.ttf','./Font Pack/SF Juggernaut Condensed Italic.ttf','./Font Pack/engebold.ttf',
'./Font Pack/BastardusSans.ttf','./Font Pack/SF Juggernaut Condensed.ttf','./Font Pack/gadugib.ttf',
'./Font Pack/Battleground.ttf','./Font Pack/SF Laundromatic Bold Oblique.ttf','./Font Pack/hawkmooncondital.ttf',
'./Font Pack/Blacklisted.ttf','./Font Pack/SF Movie Poster Bold Italic.ttf','./Font Pack/helsinki.ttf',
'./Font Pack/Bombing.ttf','./Font Pack/SF Movie Poster Bold.ttf','./Font Pack/impact.ttf',
'./Font Pack/Brushstrike trial.ttf','./Font Pack/SF New Republic.ttf','./Font Pack/jampact.ttf',
'./Font Pack/Buran USSR.ttf','./Font Pack/SF Old Republic Bold.ttf','./Font Pack/kirsty__.ttf',
'./Font Pack/Buran USSR_0.ttf','./Font Pack/SF Old Republic SC Bold Italic.ttf','./Font Pack/kr1.ttf',
'./Font Pack/CHINESER.ttf',	'./Font Pack/SF Old Republic SC Bold.ttf','./Font Pack/krazynights.ttf',
'./Font Pack/CHINESETAKEAWAY.ttf','./Font Pack/SF Speakeasy Oblique.ttf','./Font Pack/land-v2.ttf',
'./Font Pack/COUTURE-Bold.ttf','./Font Pack/SF Speakeasy.ttf','./Font Pack/later_on.ttf',
'./Font Pack/Champagne & Limousines Bold.ttf','./Font Pack/SF Sports Night NS Upright.ttf','./Font Pack/libelsuit.ttf',
'./Font Pack/ClearLine_PERSONAL_USE_ONLY.ttf','./Font Pack/SF Sports Night NS.ttf','./Font Pack/ltromatic bold.ttf',
'./Font Pack/Colors Of Autumn.ttf','./Font Pack/SF Toontime B Italic.ttf','./Font Pack/lucid.ttf',
'./Font Pack/Comic Book.ttf','./Font Pack/SF Toontime B.ttf','./Font Pack/mangat.ttf',
'./Font Pack/Comic Book.ttf','./Font Pack/SF Toontime B.ttf','./Font Pack/mangat.ttf',
'./Font Pack/Comicv2b.ttf',	'./Font Pack/SF Willamette Bold Italic.ttf','./Font Pack/medium.ttf',
'./Font Pack/Comicv2bi.ttf','./Font Pack/SF Willamette Bold.ttf','./Font Pack/medra___.ttf',
'./Font Pack/Comicv2c.ttf','./Font Pack/SF Willamette.ttf','./Font Pack/nerdproof.ttf',
'./Font Pack/Copper Canyon WBW.ttf','./Font Pack/SamdanCondensed.ttf','./Font Pack/nevis.ttf',
'./Font Pack/Crimes Times Six.ttf','./Font Pack/Schluber.ttf','./Font Pack/novem___.ttf',
'./Font Pack/DHF Story Brush.ttf','./Font Pack/SelznickNormal.ttf','./Font Pack/overhead.ttf',
'./Font Pack/DK Face Your Fears.ttf','./Font Pack/SkarpaLt.ttf','./Font Pack/pakenham.ttf',
'./Font Pack/Dark Ministry.ttf','./Font Pack/Slim Extreme.ttf','./Font Pack/panicbuttonbb_ital.ttf',
'./Font Pack/Deadly Inked.ttf','./Font Pack/Slimania Bold.ttf','./Font Pack/panicbuttonbb_reg.ttf',
'./Font Pack/DistTh__.ttf','./Font Pack/Snowstorm Kraft.ttf','./Font Pack/pricedown bl_0.ttf',
'./Font Pack/Florsn02.ttf','./Font Pack/Snowstorm Kraft_0.ttf','./Font Pack/ratio___.ttf',
'./Font Pack/Forque.ttf','./Font Pack/Something Strange.ttf','./Font Pack/segoeuib.ttf',
'./Font Pack/Freshman.ttf','./Font Pack/Sparkler-demo.ttf','./Font Pack/seriously.ttf',
'./Font Pack/Friday13v12.ttf','./Font Pack/Square.ttf','./Font Pack/slender.ttf',
'./Font Pack/Gotham Nights.ttf','./Font Pack/Streamster.ttf','./Font Pack/spacefr.ttf',
'./Font Pack/GrutchGrotesk-CondensedLight.ttf','./Font Pack/TalkingToTheMoon.ttf','./Font Pack/spacefri.ttf',
'./Font Pack/HACKED.ttf','./Font Pack/Tall Film Fine.ttf','./Font Pack/squitcher.ttf',
'./Font Pack/HONEJ___.ttf','./Font Pack/Tall Film Oblique.ttf','./Font Pack/srgt6pack.ttf',
'./Font Pack/Harabara.ttf','./Font Pack/Tall Film.ttf','./Font Pack/steelfib.ttf',
'./Font Pack/HoneyScript-SemiBold.ttf','./Font Pack/Tall Films Expanded Oblique.ttf','./Font Pack/steelfis.ttf',
'./Font Pack/Hursheys.ttf','./Font Pack/Tall Films Expanded.ttf','./Font Pack/stentiga.ttf',
'./Font Pack/IRON MAN OF WAR 002 NCV.ttf','./Font Pack/Tall Films Fine Oblique.ttf',
'./Font Pack/Jedi.ttf','./Font Pack/TequilaSunrise-Regular.ttf','./Font Pack/talldark.ttf',
'./Font Pack/KataBidalanBold.ttf','./Font Pack/Timeline.ttf','./Font Pack/telegrafico_by_ficod.ttf',
'./Font Pack/Korner Deli NF.ttf','./Font Pack/True Lies.ttf','./Font Pack/thin.ttf',
'./Font Pack/LT Oksana Bold.ttf','./Font Pack/Ubuntu-B.ttf','./Font Pack/tussle.ttf',
'./Font Pack/Laser Rod.ttf','./Font Pack/UltraCondensedSansSerif.ttf','./Font Pack/updik___.ttf',
'./Font Pack/LeelaUIb.ttf','./Font Pack/Unrealised.ttf','./Font Pack/upheavtt.ttf',
'./Font Pack/LeviReBrushed.ttf','./Font Pack/Untidy Italic Skrawl.ttf','./Font Pack/vantage.ttf',
'./Font Pack/Long_Shot.ttf','./Font Pack/Vanadine Bold.ttf',
'./Font Pack/MankSans-Medium.ttf','./Font Pack/Vegetable.ttf','./Font Pack/verdanab.ttf',
'./Font Pack/Mathematical Pi 1.ttf','./Font Pack/Waukegan LDO Black Oblique.ttf','./Font Pack/vikingsquadboldital.ttf',
'./Font Pack/Minecraftia-Regular.ttf','./Font Pack/Waukegan LDO Black.ttf','./Font Pack/vikingsquadcond.ttf',
'./Font Pack/Monotoon KK.ttf','./Font Pack/Wolf in the City Light.ttf','./Font Pack/visitor2.ttf',
'./Font Pack/Moon Flower Bold.ttf','./Font Pack/Write2.ttf','./Font Pack/viva01.ttf',
'./Font Pack/Moon Flower.ttf','./Font Pack/YanoneKaffeesatz-Bold.ttf','./Font Pack/waltographUI.ttf',
'./Font Pack/Moscoso.ttf','./Font Pack/YanoneKaffeesatz-Light.ttf','./Font Pack/weaver.ttf',
'./Font Pack/Mouser.ttf', './Font Pack/YanoneKaffeesatz-Regular.ttf','./Font Pack/webster.ttf',
'./Font Pack/NFS_by_JLTV.ttf','./Font Pack/YanoneKaffeesatz-Thin.ttf','./Font Pack/wunderbar.ttf',
'./Font Pack/NHL Ducks.ttf'	,'./Font Pack/Yearsupplyoffairycakes.ttf','./Font Pack/wyldb.ttf',
'./Font Pack/Nirmala.ttf','./Font Pack/Yellowc.ttf','./Font Pack/wyldt.ttf',
'./Font Pack/NirmalaB.ttf','./Font Pack/absci___.ttf','./Font Pack/yesterdaysmeal.ttf',
'./Font Pack/ObelixProB-cyr.ttf','./Font Pack/abscib__.ttf','./Font Pack/yonder.ttf',
'./Font Pack/OpenSans-Semibold.ttf','./Font Pack/accid__.ttf','./Font Pack/yukari.ttf',
'./Font Pack/OperationalAmplifier.ttf','./Font Pack/aggstock.ttf','./Font Pack/zeldadxt.ttf',
'./Font Pack/Opificio.ttf','./Font Pack/annifont.ttf','./Font Pack/zephyrea.ttf',
'./Font Pack/Opificio_Bold.ttf']

characters = list(string.ascii_letters[0:26]) #+list(string.digits)
print (characters)
Y = []
count = 0
for i,text in enumerate(characters):
    for font_i in fonts:
        font = ImageFont.truetype(font_i, 18)
        max_size = font.getsize(text)
        offset = (7,5)
        max_size = (28,28)
        name = str(text)+str(count)+'.png'
        count = count+1        
        MakeImg(text, font, name, max_size, offset)
        Y.append(name +','+str(i)+','+text)

#Write CSV file
with open('Train.csv', 'w') as F:
    F.write('image,label,character\n')
    F.write('\n'.join(Y))
