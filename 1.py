from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
import io
import time
import os
import shutil
from kivy.core.image import Image as CoreImage
import sys
sys.path.append(r'src/POLYX')
sys.path.append(r'src/DVR')
from recgcolor import box_POLYX
from crop import box_DVR
import cv2
import kivy
kivy.require('1.8.0')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
class LoadDialogInfo(FloatLayout):
    """docstring for LoadDialogInfo"""
    LOG_info = StringProperty()
    cancel = ObjectProperty(None)
class RootWidget(BoxLayout):
    '''Create a controller that receives a custom widget from the kv lang file.
    Add an action to be called from a kv file.
    '''

    boxName = StringProperty()
    imgPath = StringProperty()
    nailNum = StringProperty()
    loadfile = ObjectProperty(None)
    LOG_info = StringProperty()
    cap = ObjectProperty()
    take_pic = BooleanProperty()
    temp_pic = StringProperty()
    global mouseimg, point1, point2
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_info(self):
        content = LoadDialogInfo(LOG_info= self.LOG_info,cancel=self.dismiss_popup)
        self._popup = Popup(title="Tips",content = content,
                            size_hint=(0.4, 0.4),font_size = '16px')
        self._popup.open()

    def show_load(self,text):
        # if len(text)>10:
        #     self.LOG_info = 'Select a box first!'
        #     self.show_info()
        # else:
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
        # self.imgPath = './img/green.jpg'

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.imgPath = os.path.join(path, filename[0])
        self.dismiss_popup()

    def do_action(self, boxname):
        # self.show_box.text = boxname
        self.boxName = boxname

    def count(self,imgPath, boxname):

        if imgPath == '' or boxname == '' :
            self.LOG_info = "Image is Null \n Or Box is Null"
            self.show_info()
            counting = self.ids['counting']
            counting.state = 'normal'
        else:
           
            img = cv2.imread(imgPath)
            if(boxname=='POLYX-green' or boxname=='POLYX-yellow'):
                color = boxname[6:]
                self.nailNum , _= box_POLYX(img,color)
            if(boxname=='DVR'):
                self.nailNum = box_DVR(img,boxname)
            if(boxname=="ALPS"):
                pass
    
    def on_mouse(self, event, x, y, flags, param):
        global mouseimg, point1, point2
        img2 = mouseimg.copy()
        if event == cv2.EVENT_LBUTTONDOWN: 
            point1 = (x,y)
            cv2.circle(img2, point1, 10, (0,255,0), 5)
            cv2.imshow('capture', img2)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
            cv2.imshow('capture', img2)
        elif event == cv2.EVENT_LBUTTONUP: 
            point2 = (x,y)
            cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
            cv2.imshow('capture', img2)
            min_x = min(point1[0],point2[0])     
            min_y = min(point1[1],point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] -point2[1])
            cut_img = mouseimg[min_y:min_y+height, min_x:min_x+width]
            self.save_img(cut_img)
            self.take_pic = True
    
    def Onplay(self):
        global mouseimg, take_pic
        cap = cv2.VideoCapture(0)
        self.take_pic = False
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
        while (1):
            ret, frame = cap.read()
            cv2.imshow('capture',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                mouseimg = frame.copy()
                print("Captured")
                break
        cap.release()
        self.save_img(mouseimg)
        cv2.setMouseCallback('capture', self.on_mouse)
        while (self.take_pic==True):
            cv2.destroyAllWindows()
        print(self.take_pic)

    def save_img(self, img):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        path = "temp/IMG_{}.png".format(timestr)
        cv2.imwrite(path, img)
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        self.imgPath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),path)    

Factory.register('LoadDialog', cls=LoadDialog)

class CountNailApp(App):

    '''This is the app itself'''

    def setDir(self):
        if self.temp_pic not in os.listdir('./'):
            os.mkdir('./'+self.temp_pic)
        else:
            shutil.rmtree('./'+self.temp_pic)
            os.mkdir('./'+self.temp_pic)

    
    def build(self):
        '''This method loads the root.kv file automatically

        :rtype: none
        '''
        # loading the content of root.kv
        self.root = Builder.load_file('kv/root.kv')
        self.temp_pic = 'T_pic'
        # self.setDir()


    def next_screen(self, screen):
        '''Clear container and load the given screen object from file in kv
        folder.

        :param screen: name of the screen object made from the loaded .kv file
        :type screen: str
        :rtype: none
    '''

        filename = screen + '.kv'
        # unload the content of the .kv file
        # reason: it could have data from previous calls
        Builder.unload_file('kv/' + filename)
        # clear the container
        self.root.container.clear_widgets()
        # load the content of the .kv file
        screen = Builder.load_file('kv/' + filename)
        # add the content of the .kv file to the container
        self.root.container.add_widget(screen)


if __name__ == '__main__':
    CountNailApp().run()
    
