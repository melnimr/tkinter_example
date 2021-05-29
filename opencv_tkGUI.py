from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text
from PIL import Image,ImageTk
import pytesseract as pt
from pytesseract import Output
import cv2
import numpy as np
import random as rng

from math import floor


counter = 0
   

def get_shape_contour(contours):    
    # loop over the contours
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True) # 0.032
        if len(approx) == 4  and  cv2.isContourConvex(approx):
            return approx
def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio
def warp_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


def choose_file():

    global filepath, img_resize,resize_ratio,original
    filepath = filedialog.askopenfilename(initialdir = 'E:\\',title = 'Select an Image',filetypes = (('JPG','*.jpg'),('All files','*.*')))
    img = cv2.imread(filepath)
    resize_ratio = 500 / img.shape[0]
    original = img.copy()
    img_resize = opencv_resize(img,resize_ratio)
    cv2.imshow('Image', img_resize)
    print("::Running choose file")
    
def autocrop():
    global transform1,final_image
    
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    #canny = cv2.Canny(img_thresh,150, 300)


    canny = cv2.Canny(dilated, 100,200, apertureSize=3)
    canny = cv2.dilate(canny, None )
    print("::Running AutoCrop")

    contours, heirarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    receipt_contour = get_shape_contour(largest_contours)
    final_image = warp_perspective(original.copy(), contour_to_rect(receipt_contour))
    print(final_image.shape)
    cv2.imshow('Transformed', final_image)
    cv2.waitKey(0)
    

    return(final_image)



def largest_contour():
    global image_with_contour
    
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    #rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    rectKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    #canny = cv2.Canny(img_thresh,150, 300)
    canny = cv2.Canny(dilated, 100,200, apertureSize=3)
    print("::Running largest contour")
    
    #contours, heirarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, heirarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    

    for i in range(len(largest_contours)):
        x, y, w, h = cv2.boundingRect(i)
        cv2.rectangle(canny, (x, y), (x+w, y+h), (255, 255, 255), -1)
    cv2.drawContours(canny, hull_list, -1, (255,0,0),3)
    #cv2.imshow('test canny', canny)
    contours, heirarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    for c in largest_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True) # 0.032
        if len(approx) == 4 and cv2.isContourConvex(approx):
            image_with_contour = cv2.drawContours(img_resize.copy(),approx , -1 , (0,255,0),3)
            break
    cv2.imshow('Largest contour', image_with_contour)
    cv2.waitKey(0)
def largest_contours():
    global transform1, image_with_contours
    
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    #canny = cv2.Canny(img_thresh,150, 300)
    canny = cv2.Canny(dilated, 100,200, apertureSize=3)
    print("::Running largest contours")
    
    #contours, heirarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, heirarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    hull_list = []
    for i in range(len(largest_contours)):
        hull = cv2.convexHull(largest_contours[i])
        hull_list.append(hull)
    #for ii in range(len(largest_contours)):
    image_with_contours = img_resize.copy()
    ii = -1
    print(len(hull_list))
    cv2.drawContours(image_with_contours,largest_contours , ii, (0,255,0),3)
    cv2.drawContours(image_with_contours,hull_list , ii, (255,0,0),3)
    cv2.imshow('Largest ten contours', image_with_contours)
    cv2.waitKey(0)

def grabcut():

    mask = np.zeros(output_image.shape[:2],dtype = np.uint8)
    rect = (0,0,1,1)
    y = output_image.shape[0] - 100
    x = output_image.shape[0] - 100
    rect = (100,100,y,x)
    grab_cut = output_image.copy()
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    cv2.grabCut(grab_cut,mask,rect,bgdmodel,fgdmodel,1, cv2.GC_INIT_WITH_RECT)
    cv2.imshow('GrabCut',grab_cut)
    cv2.waitKey(0)
    
def ocr():

    pt.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    global data

  
    if autocrop() is None:
        img = mancrop()
    
    elif mancrop() is None:
        img = autocrop()

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    ret,img_thresh = cv2.threshold(img_blur,170,300,cv2.THRESH_BINARY)

    data =  pt.image_to_string(img_thresh)
    

    text = pt.image_to_data(img_thresh, output_type=Output.DICT)

    no_word = len(text['text'])

    for i in range(no_word):
        if int(text['conf'][i]) > 50:
            x,y,width,height = text['left'][i], text['top'][i], text['width'][i], text['height'][i]
            cv2.rectangle(img_resize, (x,y), (x+width, y+height), (0,255,0), 2)
            cv2.imshow('OCR-operated',img_resize)
            
def showtext():
    content = data
    textbox = tk.Frame(frame,bg = '#FDFFD6')
    textbox.place(relx = 0.2,rely = 0.2,relwidth =0.6,relheight =0.6)
    text_frame = Text(textbox,bg = '#FDFFD6')
    text_frame.insert('1.0',content)
    text_frame.pack()

def circle_crop():
    global output_image
    
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(gray, 45)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50 ,param2 = 30, minRadius = 0, maxRadius =0 )
    (x,y,r) = circles[0,0,:]

    height, width = image.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    circle_img = cv2.circle(mask, (x,y), int(r), (255,255,255), thickness = -1)
    masked_data = cv2.bitwise_and(final_image, final_image, mask = circle_img)
    x,y,w,h = cv2.boundingRect(mask)
    masked_data = masked_data[y:y+h, x:x+w]
    h,w = masked_data.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[:] = 0
    lo = 30
    hi = 100
#    flags = cv2.FLOODFILL_FIXED_RANGE
 #   cv2.floodFill(masked_data, mask, (floor(w/2),floor(h/2)), (255, 255, 255), (lo,)*3, (hi,)*3, flags)

    masked_data_new = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(masked_data_new, (5,5), 0 )
    edges = cv2.Canny(blurred, 0, 200)    
    contours, heirarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    output_image = masked_data.copy()
    hull = []
    bounding_rect = []
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        h= cv2.convexHull(contours[i], False)
        area = cv2.contourArea(h)
        approx = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(approx)
       
        if area < 10000:
            hull.append(h)
            cv2.rectangle(output_image , (int(boundRect[0]), int(boundRect[1])),
                          (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), color, 2)
            #bounding_rect.append(bc)


    #cv2.drawContours(output_image, bounding_rect, -1, (0,0,0), 2)  
    # Find Circles
   # circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT, 1, 10
   #                            , param1 = 50, param2 = 30,
  #                             minRadius = 10, maxRadius = 200)
  #  sorted_circles = sorted(circles[0], key = lambda x:x[2], reverse = True)
  #  circles = np.uint16(np.around(circles))  
  #  output_image = masked_data.copy()
  #  for i in circles[0,:]:
  #      cv2.circle(output_image, (i[0], i[1]), i[2], (0,255,0) , 2)
    cv2.imshow('Circle Crop', output_image)
    cv2.waitKey(0)

def save():
    global counter
    global image_resize
    counter+=1
    cv2.imwrite('image_'+str(counter) + '.jpg', my_image)
def exit_app():
    root.quit()

root = tk.Tk()

canvas = tk.Canvas(root,height = 800,width = 800,bg = 'green') 
canvas.pack()

frame = tk.Frame(root,bg = 'white')
frame.place(relx = 0.2,rely = 0.1,relwidth =0.6,relheight =0.6)

label=tk.Label(frame,text='TEXT DETECTOR   ',fg='green',bg='white',font=('Arial',20))
label.place(relx=0.28,rely=0.1)


openfile = tk.Button(canvas,text = 'Choose a File',fg = 'blue',padx = 10,pady = 5,command = choose_file )
openfile.place(x = 50 , y = 600)

auto_crop = tk.Button(canvas,text = 'Auto Crop',fg = 'green',padx = 10,pady = 5,command = autocrop)
auto_crop.place(x = 370 ,y = 600)

manual_crop = tk.Button(canvas,text = 'GrabCut',fg = 'purple',padx = 10,pady = 5,command = grabcut)
manual_crop.place(x = 215 ,y = 600)


OCR_btn = tk.Button(canvas,text = '    L/C   ',fg = 'brown',padx = 10,pady = 5, command = largest_contour)
OCR_btn.place(x = 500, y = 600)

show_txt_btn = tk.Button(canvas,text = 'Show Largest Contours',fg = 'violet',padx = 10,pady = 5, command = largest_contours)
show_txt_btn.place(x = 640, y = 600)

save_btn = tk.Button(canvas,text = 'Circle Crop',fg = 'orange',padx = 10,pady = 5, command = circle_crop)
save_btn.place(x = 50, y = 700)

exit_btn = tk.Button(canvas,text = 'Exit',fg = 'red',padx = 10,pady = 5, command = exit_app)
exit_btn.place(x = 700, y = 700)

root.mainloop()
