from runningmodel import calculation_module
from runningmodel_alpha import alpha_module
from only_number_detection import only_numbers
from pdfmail import pdfgone
import glob
import os



files=glob.glob("C:/Users/arindam bhattacharya/Documents/hkhack/answerphotos/*")
for f in files:
    os.remove(f)
key=True
while(key==True):
    command=input('enter what to do:')
    if(command=='number'):
        only_numbers()
    if(command=='calculation'):
        calculation_module()
    if(command=='english'):
        alpha_module()
    if(command=='exit'):
        pdfgone()
        key=False
