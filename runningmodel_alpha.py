import os
import PIL
import cv2
import glob
from tkinter import *
from PIL import Image,ImageDraw
import pyscreenshot as ImageGrab
import matplotlib.pyplot as plt
import win32gui
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import pandas as pd
from check_correct_answer_alpha import right_answer_alpha
from rightpopup import right_pop
from wrongpopup import wrong_pop
#from pdfmail import pdfgone
#image_number=0
#right_answer=''


#def alpha_module():

json_file = open('model_alpha.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights("model_alpha.h5")
print("Loaded model from disk")
right_answer=input('Teachers question')
files=glob.glob("C:/Users/arindam bhattacharya/Documents/hkhack/answerphotos/*")
for f in files:
   os.remove(f)
letters_num = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
             7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
             14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
             21:'V',22:'W',23:'X', 24:'Y',25:'Z'}



root=Tk()
root.resizable(0,0)
root.title('Handwritten  recognition GUI App')
#model.load('model.h5')
#######
lastx,lasty=None,None
image_number=0
#canvas
cv=Canvas(root,width=1000,height=300,bg='white')#change the width to adjust no of digits
cv.grid(row=0,column=0,pady=2,sticky=W,columnspan=2)
########

########

#######


#canvas function implementation
def clear_widget():
    global cv
    cv.delete('all')
    global right_answer
    right_answer=input('Teachers question')

def activate_event(event):
    global lastx,lasty
    #<B1-Motion>
    cv.bind('<B1-Motion>',draw_lines)
    lastx,lasty=event.x,event.y
def draw_lines(event):
    global lastx,lasty
    x,y=event.x,event.y
    #drawing
    cv.create_line((lastx,lasty,x,y),width=8,fill='black',capstyle=ROUND,smooth=TRUE,splinesteps=12)
    lastx,lasty=x,y
#recognizing digit
def Recognize_Digit():
    global image_number
    prediction=[]
    precentage=[]
    xa=[]
    ya=[]
    filename=f'image_{image_number}.png'
    widget=cv
    #co-ordinates
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)
    #HWND = cv.winfo_id() # get the handle of the canvas
    #rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
    #ImageGrab.grab(rect).save(filename)
    image=cv2.imread(filename,cv2.IMREAD_COLOR)
    #cv2.imread(image,image)
    #plt.imshow(image)
    gray=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    ret,th=cv2.threshold(gray.copy(),0,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #preprocessed_digits=[]
    #plt.imshow(contours)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    #print(contours)
    
    preprocessed_digits=[]
    for cnt in contours:
        #get bounding box and exact ROI
        
        
        x,y,w,h=cv2.boundingRect(cnt)
        #create rectangle
        cv2.rectangle(image,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        top=int(0.05*th.shape[0])
        bottom=top
        left=int(0.05*th.shape[1])
        right=left
        th_up=cv2.copyMakeBorder(th,top,bottom,left,right,cv2.BORDER_REPLICATE)
        th_up=th[y:y+h,x:x+w]
        #extract image ROI
        #plt.imshow(th_up)
        #roi=th[x-left:x+w+right,y-top:y+h+bottom]
        #roi = th[y:y+h, x:x+w]
        #plt.imshow(th_up)
        #plt.imshow(roi)
        #resize roi image
        img=cv2.resize(th_up,(18,18),interpolation=cv2.INTER_AREA)
        #padding
        img = np.pad(img, ((5,5),(5,5)),"constant", constant_values=0)

        #reshaping the image
        img=img.reshape(1,28,28,1)
        
        img=img/255.0
        preprocessed_digits.append(img)
        xa.append(x)
        ya.append(y)
       
    i=0
    for digit in preprocessed_digits:
        pred=model1.predict(digit)[0]
        ######
        final_pred=np.argmax(pred)
        data=str(letters_num[final_pred])+' '+str(int(max(pred)*100))+'%'
        #draw text on image
        font=cv2.FONT_HERSHEY_SIMPLEX
        fontScale=0.5
        color=(255,0,0)
        thickness=1
        cv2.putText(image,data,(xa[i],ya[i]),font,fontScale,color,thickness)
        i=i+1
        cv2.imshow('image',image)
        cv2.waitKey(0)
        x=x+x
    #print(final_pred)
        print(pred)
        print(np.argmax(pred))
        prediction.append(str(letters_num[final_pred]))
    print("reconized digit: "+str(prediction))
    global right_answer
    key=right_answer_alpha(right_answer,prediction)
    if key==True:
        right_pop()
    else:
        wrong_pop()
        
    
#showing the thing

cv.bind('<Button-1>',activate_event)
btn_save=Button(text='recognize',command=Recognize_Digit)
btn_save.grid(row=2,column=0,pady=1,padx=1)
button_clear=Button(text='clear widget',command=clear_widget)
button_clear.grid(row=2,column=1,pady=1,padx=1)
#mainloop
root.mainloop()
#pdfgone()

