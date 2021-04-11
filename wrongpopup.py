from tkinter import * 
import tkinter.messagebox
import pyautogui 
import random
import string
root=Tk() 
def wrong_pop():
    letters = string.ascii_lowercase
    a=''
    a.join(random.choice(letters))
    a=a+a.join(random.choice(letters) for i in range(5))

    tkinter.messagebox.showinfo('box','wrong!')
    #print(a)
    myScreenshot=pyautogui.screenshot()
    myScreenshot.save(r"C:\Users\arindam bhattacharya\Documents\hkhack\answerphotos\sc" +str(a)+".png")
    print('screencaptured')
    root.mainloop()
    