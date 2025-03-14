""" A short program to rename the files in this folder from,
for example, 'lecture0_page-0001.jpg' to 'CS50x_Lecture0_Slide-001.jpg' """
import os
for filename in os.listdir("."):
    if filename.endswith(".jpg"):
        #print(filename)
        os.rename(filename, f"CS50x_Lecture{filename[7]}_Slide-{filename[-7:-4]}.jpg")
