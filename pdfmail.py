import glob 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
from PIL import Image



def pdfgone():
    pdf=FPDF()
    files=glob.glob("C:/Users/arindam bhattacharya/Documents/hkhack/answerphotos/*")
    pdf.add_page()
    
    for image in files:
        im=Image.open(image)
        reim = im.resize((round(im.size[0]*0.3), round(im.size[1]*0.5)))
        reim.save(image)
        pdf.image(image)
    pdf.output("today_work.pdf")
 
    
    mail_content = '''Hello,
    Today's work done in class by student x
    Thank You
    '''
    
    sender_address = 'gmeethelpbot@gmail.com'
    sender_pass = '**********'
    receiver_address = 'arindam01012001@gmail.com'
    
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Automated mail for todays class work .'
    #The subject line
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    attach_file_name = 'work.pdf'
    attach_file = open('today_work.pdf', 'rb') # Open the file as binary mode
    payload = MIMEBase('application', 'pdf', Name='work.pdf')
    payload.set_payload((attach_file).read())
    encoders.encode_base64(payload) #encode the attachment
    #add payload header with filename
    payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
    message.attach(payload)
    
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')