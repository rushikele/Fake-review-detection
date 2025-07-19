

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk

##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Fake review detection Using ML")

# 43
# video_label =tk.Label(root)
# video_label.pack()
#  read video to display on label
# player = tkvideo("acci.mp4", video_label,loop = 1, size = (w, h))
# player.play()
# ++++++++++++++++++++++++++++++++++++++++++++
####For background Image
image2 = Image.open('img3.jpg')
image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=70)  # , relwidth=1, relheight=1)

label_l1 = tk.Label(root, text="Product Review Sentiment Analysis Using ML",font=("Times New Roman", 30, 'bold'),
                    background="#000000", fg="white", width=70, height=2)
label_l1.place(x=0, y=0)

#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def cap_video():
    
#     video1.upload()
#     #from subprocess import call
#     #call(['python','video_second.py'])

def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
    root.destroy()
    
def Sarcasm():
    from subprocess import call
    call(["python","Sarcasm_Detector.py"])
        
    
    
def window():
  root.destroy()


button1 = tk.Button(root, text="LOGIN", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="#FFEBCD", fg="brown", bd=5, relief='ridge')
button1.place(x=20, y=190)

button2 = tk.Button(root, text="REGISTER",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="#FFEBCD", fg="brown", bd=5, relief='ridge')
button2.place(x=20, y=300)

button3 = tk.Button(root, text="EXIT",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="red", fg="black", bd=5, relief='ridge')
button3.place(x=20, y=450)
root.mainloop()