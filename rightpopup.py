from tkinter import * 
import tkinter.messagebox
import  pyautogui
import random
import string
root=Tk() 
def right_pop():
    letters = string.ascii_lowercase
    a=''
    a=a+a.join(random.choice(letters) for i in range(5))
    #print(a)
    tkinter.messagebox.showinfo('box','Correct!')
    myScreenshot=pyautogui.screenshot()
    path='C:/Users/arindam bhattacharya/Documents/hkhack/answerphotos/sc'+str(a)+str('.png')
    myScreenshot.save(path)
    print('screencaptured')
    root.mainloop()