# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:19:41 2019

@author: prasad Kevin
"""


import numpy as np
from PIL import Image, ImageEnhance
import cv2, glob, os, ntpath, csv
import random, sys
import utils
from misc import (get_all_subdirs, mkdir_p, getImageFilesInFold,find_enclosing_rect, rotate_bound)
#import pandas as pd
##################### To create folder ###############





def createDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
############### To remove the path name ##################
def path_leaf(path):
    
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
################# Defining the output frame and identifing the object ##############
def display_output(image, bbox, name):
    
    for x_min, y_min, x_max, y_max in bbox:
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        
        ############################################### condition    
   
    
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
################################ saving the image with count and name ##############
def save_images(image, boxes, name, count, obj_classes):
    
    if cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) == 0:     # if a frame is blank then do nothing.
        print("<<<<<<<<<<<<< blank imageeeeeeee >>>>>>>>>>>>>>")
        return
    
    file_name = str(count) + '_' + name + '_img.jpg'
    image_path = with_objects + file_name
    
    for i in range(len(boxes)):
        
    #    print(obj_classes[i])
        x_min, y_min, x_max, y_max = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
    
        if x_min < 0:
            print("x_min error: ", x_min, file_name)
            x_min = 0
            
        if y_min < 0:
            print("y_min error: ", y_min, file_name)
            y_min = 0
            
        if x_max > 256:
            print("x_max error: ", x_max, file_name)
            x_max = 256
            
        if y_max > 256:
            print("y_max error: ", y_max, file_name)
            y_max = 256
        
        row = [file_name, str(bg_w), str(bg_h), obj_classes[i], str(x_min), str(y_min), str(x_max), str(y_max)]
        csv_writer.writerow(row)
        
    cv2.imwrite(image_path, image)
    
    display_output(image.copy(), boxes, "boxes")   
########################### adding sunlight effect to the image ##################   
def add_sun_light(normal_image):
    ############ REAFDING th refile with randowm randit module function
    
    sun_image = cv2.imread(sun_png_files[random.randint(0, len(sun_png_files)-1)], cv2.IMREAD_UNCHANGED)
    places = np.where(sun_image[:,:, 3] == 255)
    alpha_layer = sun_image[:, :, 3]
    alpha_layer[places] = 120
    sun_image[:,:, 3] = alpha_layer  
    sun_image = rotate_and_crop(sun_image,  random.uniform(0, 360), pix_border=0)
    sun_image = cv2.resize(sun_image, (resize_val_w, resize_val_h))
    
    return alphaBlend(sun_image, normal_image).astype(np.uint8)
################### to change the brightness of the images
def change_brightness(img, beta):

    enhancer = ImageEnhance.Brightness(img)
    enhanced_im = enhancer.enhance(beta)
    
    return enhanced_im

noises = ["gauss", "s&p"]
##################### To add the noise to the image
def noisy(noise_typ, image):
  if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = .5
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = .5
      amount = 0.02
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      out[coords] = 255
    
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
  elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
  elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

############################# To get the ratio of the ####################################
def get_size_ratios(obj_class, im):
    
    
    if obj_class == 'child':
        
        resize_w = int(round((random.uniform(3, 4)/10) * 256, 0))
        resize_h = resize_w* (random.uniform(1.5, 2))       
        
    elif obj_class == 'Person':

        resize_w = int(round((random.uniform(3.5, 5)/10) * 256, 0))
        resize_h = resize_w*(random.uniform(1.8, 2))                
            
    elif obj_class == 'baby':
        
        resize_w = int(round((random.uniform(3, 3.5)/10) * 256, 0))
        resize_h = resize_w*(random.uniform(1.2,1.5))
        
    elif obj_class == 'Suitcase':
        
        if(im.width < im.height/1.2):
            resize_w = int(round((random.uniform(4.5, 5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(1.15, 1.35)
        else:
            resize_h = int(round((random.uniform(4.5, 5)/10) * 256, 0))
            resize_w = resize_h*random.uniform(1.15, 1.35)                    

    elif obj_class == 'Backpack':
        
        if(im.width < im.height/1.2):
            resize_w = int(round((random.uniform(2.5, 4.5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(1.4, 1.8)
        else:
            resize_h = int(round((random.uniform(2.5, 4.5)/10) * 256, 0))
            resize_w = resize_h*random.uniform(1.4, 1.8)            
        
    elif obj_class == 'Handbag':
                
        if(im.width < im.height/1.2):
            resize_w = int(round((random.uniform(2.5, 4.5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(1.15, 1.25)
        else:
            resize_h = int(round((random.uniform(2.5, 4.5)/10) * 256, 0))
            resize_w = resize_h*random.uniform(1.15, 1.25)                    
    
    elif obj_class == 'Laptop':
    
        if(im.width < im.height/1.5):
            resize_w = int(round((random.uniform(3, 4.5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(1.3, 1.5)
        else:
            resize_h = int(round((random.uniform(3, 4.5)/10) * 256, 0))
            resize_w = resize_h*random.uniform(1.6, 1.8)
    
    elif obj_class == 'Bottle':
        
        if(im.width < im.height):
            resize_w = int(round((random.uniform(1.5, 2.5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(2.75, 3.5)
        else:
            resize_h = int(round((random.uniform(1.5, 2.5)/10) * 256, 0))
            resize_w = resize_h*random.uniform(2.75, 3.5)        
        
    elif obj_class == 'Phone':
     
        if(im.width < im.height):
            resize_w = int(round((random.uniform(1.2, 2.2)/10) * 256, 0))
            resize_h = resize_w*random.uniform(1.85, 2.1)
        else:
            resize_h = int(round((random.uniform(1.2, 2.2)/10) * 256, 0))
            resize_w = resize_h*random.uniform(1.85, 2.1  )                    
    
    elif obj_class == 'Wallet':
                                                                                                                              
        if(im.width/1.8 > im.height):
            resize_w = int(round((random.uniform(1.5, 2.5)/10) * 256, 0))
            resize_h = resize_w*random.uniform(.5, .6) 
        else:
            resize_h = int(round((random.uniform(2, 2.3)/10) * 256, 0))
            resize_w = resize_h*random.uniform(.9, 1.1)                
        
    elif obj_class == 'Watch':
                        
        if(im.width < im.height/2):
            resize_w = int(round((random.uniform(1.2, 1.8)/10) * 256, 0))
            resize_h = resize_w*random.uniform(.9, 1.15)
                                
        else:
            resize_h = int(round((random.uniform(1.2, 1.8)/10) * 256, 0))
            resize_w = resize_h*random.uniform(.9, 1.15)                     


    return resize_w, resize_h

######################## overlaping the one image on other ##############
    
def alphaBlend(foreGroundImage, background):

    # Split png foreground image
    b,g,r,a = cv2.split(foreGroundImage)
    
    # Save the foregroung RGB content into a single object
    foreground = cv2.merge((b,g,r))
    
    # Save the alpha information into a single Mat
    alpha = cv2.merge((a,a,a)) 
    
    H, W = foreGroundImage.shape[:2]
    background = cv2.resize(background, (W, H))
    
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)/255
    
    # Perform alpha blending
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    
    return outImage
############################# 
def remove_spaces(orig_image):
    
    image = orig_image.copy()
    blank = (np.ones(image[:, :, :3].shape)*255).astype(np.uint8)
    image = alphaBlend(image, blank).astype(np.uint8)
    
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    
    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
  
    dst = orig_image[y:y+h, x:x+w]
    return dst
###################### changing the image
def rotate_and_crop(image, angle, pix_border = 10):
    img_rot = rotate_bound(image, angle)

    mask = img_rot > 0
    mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
    r = find_enclosing_rect(mask, pix_border)
    img_rot = img_rot[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
    rand_val = 0.9 + random.random()/5.0
    rand_val = int(rand_val*100)/100.0
    img_rot = img_rot * rand_val
    img_rot = np.clip(img_rot, 0, 255) 
    img_rot = img_rot.astype('uint8')
    
    return img_rot

######################## finiding the overlaping area
def calc_overlap(boxes):
    
    if len(boxes) < 2:
        return False
    
    box1 = boxes[0]
    box2 = boxes[1]
    
#    print(box1)
 #   print(box2)
    
    overlap_area_width =  min(box1[2], box2[2]) - max(box1[0], box2[0]) 
    overlap_area_height =  min(box1[3], box2[3]) - max(box1[1], box2[1])
    
  #  print(box1, box2)
#    print("area_width: {}, area_height: {}".format(overlap_area_width, overlap_area_height))


    if(overlap_area_width>=0 and overlap_area_height >=0):
        
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        max_area = min(area_box1, area_box2)
        overlap_area = overlap_area_width * overlap_area_height

#        print("ground truth and predicted bounding box........")
#        print("overlap_area: {}, area_box1: {}, area_box2: {}".format(overlap_area, area_box1, area_box2))
#        print("overlap: {}".format((overlap_area/max_area)))
#        
        if (overlap_area/max_area)> .3 or (overlap_area == min(area_box1, area_box2)):
            return True
        else:
            return False
        
    return False

#################### random crooping the image 
def crop_bg_randomly(bg):
   
   width, height = bg.size
   padding = 400
   top_left_x = random.randint(0, width-padding) if width-padding>0 else random.randint(0, width)
   top_left_y = random.randint(0, height-padding) if height-padding>0 else random.randint(0, height)
   border = (top_left_x, top_left_y, top_left_x+padding, top_left_y + padding)
#   print("border {}".format(border))
   return bg.crop(border)

###################### defining new created image 
def create_new_image(objects_list, r_count = 0, bg=None):
    
    count = len(objects_list)
    normal_boxes =  np.array([])
    currentlabel = []
    
    if not bg :

        
        bg = Image.open(backgrounds[random.randint(0, bg_count-1)])
    
    bg_w, bg_h = bg.size
    
    if bg_w >= 400 and bg_h >= 400:
        
        bg = crop_bg_randomly(bg)
        
    bg = bg.resize((resize_val_w, resize_val_h))
    
    bg_w, bg_h = bg.size
    
    new_imgs_list = []
    for i in range(0, count):
   #     print(i)
        im, resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class = objects_list[i]
        currentlabel.append(obj_class)
        
#        print(resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class)
#        print(im)
        if obj_class not in ["Person", "child", "baby"]:
            img_rot = rotate_and_crop(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA), random.uniform(0, 360), pix_border=0)
            img_rot = remove_spaces(img_rot)
            im = Image.fromarray(cv2.cvtColor(img_rot.copy(), cv2.COLOR_BGRA2RGBA))
        
        new_imgs_list.append(im)
        im_w, im_h = im.size
        
#        x1, y1 = int(bg_w*w_scale_factor), int(bg_h*h_scale_factor - im_h)
        
#        print(h_scale_factor, w_scale_factor)
#        h_scale_factor = h_scale_factor if (h_scale_factor<.8)  else .1
#        w_scale_factor = w_scale_factor if (w_scale_factor <.8) else .2
        
#        x1, y1 = int(bg_w*w_scale_factor), int(bg_h*h_scale_factor- (im_h // 2))
        
#        x1, y1 = int(bg_h*w_scale_factor), int(bg_w*h_scale_factor)
        
#        if i == 0:
        x1, y1 = int(bg_w*w_scale_factor - im_w), int(bg_h*h_scale_factor - im_h)
#        else:
#            x1, y1 = int(bg_w*w_scale_factor - im_w), int(bg_h*h_scale_factor - im_h)
        
#        print("count {}, x1 {}, y1 {}".format(r_count, x1, y1))
                    
        if x1 < 0:
            x_min = 0
        else:
            x_min = x1
            
        if y1 < 0:
            y_min = 0
        else:
            y_min = y1
        
        x_max = x_min + im_w
        y_max = y_min + im_h
        
        if x_max > resize_val_w:
            x_max = resize_val_w
        else:
            x_max = x_max
            
        if y_max > resize_val_h:
            y_max = resize_val_h
        else:
            y_max = y_max
                
        if normal_boxes.size ==0 :    
            normal_boxes = np.asarray([[x_min, y_min, x_max, y_max]])
        else:
            normal_boxes = np.append(normal_boxes, np.asarray([[x_min, y_min, x_max, y_max]]), axis =0)
    
    overlap = calc_overlap(normal_boxes)
    
    if not overlap:
        for i in range(0, count):
            x_min, y_min, x_max, y_max = normal_boxes[i]
            im = new_imgs_list[i]
#            print(x_min, y_min, x_max, y_max)
            bg.paste(im, (x_min, y_min), im)
        
        normal_image = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)
#        print(normal_boxes)
        return normal_image, normal_boxes, currentlabel
    
    else:
#        print("overlap exists...!!")
        
        new_objects_list = objects_list
        
        if r_count < 5 :
                
            im, resize_w, resize_h, w_scale_factor1, h_scale_factor1, obj_class = objects_list[1]
            
            w_scale_factor1 = w_scale_factor1 + w_scale_factor1*.1
            h_scale_factor1 = h_scale_factor1 + h_scale_factor1*.1
            
            h_scale_factor1 = h_scale_factor1 if (h_scale_factor1 < .9)  else .4
            w_scale_factor1 = w_scale_factor1 if (w_scale_factor1 <.9) else .4
            
            new_objects_list[1] = im, resize_w, resize_h, w_scale_factor1, h_scale_factor1, obj_class
        
        else:
            
#            print("r_count increased..!!!")
            im, resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class = objects_list[0]
            im1, resize_w1, resize_h1, w_scale_factor1, h_scale_factor1, obj_class1 = objects_list[1]
            
            w_scale_factor = .3
            h_scale_factor = .3
            
            w_scale_factor1 = .9
            h_scale_factor1 = .9
              
            new_objects_list[0] = im, resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class
            new_objects_list[1] = im1, resize_w1, resize_h1, w_scale_factor1, h_scale_factor1, obj_class1
        
        r_count += 1
        
        normal_image, normal_boxes, labels = create_new_image(new_objects_list, r_count)
        
        return normal_image, normal_boxes, labels

##################### creating blur image for the exixting image
        
def create_new_blur_image(im, w_scale_factor, h_scale_factor, obj_class, bg=None):
    
    print(im)
    if obj_class not in ["Person", "child", "baby"]:
        img_rot = rotate_and_crop(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA), random.uniform(0, 360), pix_border=0)
        img_rot = remove_spaces(img_rot)
        im = Image.fromarray(cv2.cvtColor(img_rot.copy(), cv2.COLOR_BGRA2RGBA))
    
    if not bg :
        bg = Image.open(backgrounds[random.randint(0, bg_count-1)])

    bg = bg.resize((resize_val_w, resize_val_h))

    bg_w, bg_h = bg.size
    im_w, im_h = im.size
    
#    x1, y1 = int(bg_w*w_scale_factor), int(bg_h*h_scale_factor - im_h)
    
    x1, y1 = int(bg_w*w_scale_factor - im_w), int(bg_h*h_scale_factor - im_h)
    
#    print("<<<<<<<<  >>>>>>>>>>>>>>>>>>>>", x1, y1)
    
   
    if x1 < 0:
        x_min = 0
    else:
        x_min = x1
        
    if y1 < 0:
        y_min = 0
    else:
        y_min = y1
    
    x_max = x_min + im_w
    y_max = y_min + im_h
    
    if x_max > resize_val_w:
        x_max = resize_val_w
    else:
        x_max = x_max
        
    if y_max > resize_val_h:
        y_max = resize_val_h
    else:
        y_max = y_max
    
    bg.paste(im, (x_min, y_min), im)
    normal_image = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)
    
    normal_boxes = np.asarray([[x_min, y_min, x_max, y_max]])
    
    return normal_image, normal_boxes


############### reading the image with alpha layer
def process_object(obj, h_scale_factor, w_scale_factor):
    
    obj_class = obj.split('\\')[1]
#    obj_class = obj.split('/')[1]
    print(obj_class)
    if obj_class in ['Person', 'baby']:
        im = Image.open(obj).convert("RGBA")
    else:
        im = Image.open(obj)
    
    resize_w, resize_h  = get_size_ratios(obj_class, im)
        
    resized_im = im.resize((int(resize_w), int(resize_h)))

    return [resized_im, resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class]
    
############### defining co-ordinates
def get_fourcoords():
    
    ob1_w = random.randint(4500, 7000)/10000
    ob1_h = random.randint(4500, 7000)/10000
    
    ob2_w = random.randint(0, 4000)/10000
    ob2_h = random.randint(0, 4000)/10000
    return ob1_w, ob1_h, ob2_w, ob2_h

########### obtaining four coordinates
def get_four_coords(ob1_w, ob1_h, ob2_w, ob2_h):
    
    if ob1_h <.5 and ob2_h<.5:
       ob1_h = ob1_h + .5 
       if (ob1_h>.8):
           ob1_h = .5

    if ob1_h >=.5 and ob2_h>=.5:
       ob1_h = ob1_h - .5  
    
    if ob1_w <.5 and ob2_w<.5:
       ob1_w = ob1_w + .5  
       if (ob1_w>.8):
           ob1_h = .5
       
    if ob1_w >=.5 and ob2_w>=.5:
       ob1_w = ob1_w - .5  
     
    ob1_h = ob1_h if (ob1_h<.8)  else .1
    ob1_w = ob1_w if (ob1_w <.8) else .2  

    ob2_h = ob2_h if (ob2_h<.8)  else .1
    ob2_w = ob2_w if (ob2_w <.8) else .2  

    return ob1_w, ob1_h, ob2_w, ob2_h

################ main function

if __name__ == "__main__":
    

    count = 0
    resize_val_w = 256
    resize_val_h = 256
    display_flag = False
    
    #new_background_images
    backgrounds = glob.glob('background_images/*')
       
    objects = glob.glob('objects/*/*')
#    sun_png_files = glob.glob("sun/*")
    
    with_objects = 'images/train/'
    csv_path =  'data/'
    
#    'Multiple_Car_Objects'
#    with_objects = sys.argv[1] + '/JPEGImages/'
#    csv_path = sys.argv[1] + '/csv/'
 ###################################### ???????????????   
    createDir(csv_path)
    createDir(with_objects) 
       
    csv_file = csv_path + '/train.csv'
    if os.path.exists(csv_file):
        output_file = open(csv_file,'a', newline='')
    else:
        output_file = open(csv_file,'w', newline='')
        csv_writer  = csv.writer(output_file)
        row = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
        csv_writer.writerow(row)
    csv_writer  = csv.writer(output_file)
    
    file_count = 409700

    bg_count = len(backgrounds)
    
    all_objects_count = len(objects) - 1
    
    total = 1
    
#    mu, sigma = 0, 0.5
#    x = np.random.normal(mu, sigma, 10000)
#    gauss_distribution = abs(x)
    
    for i in range(0, total):
        
    #    obj_class = obj.split('/')[1]
        
        bg = Image.open(backgrounds[random.randint(0, bg_count-1)])
        bg = bg.resize((resize_val_w, resize_val_h))
    
        objects_list = []
        print("\n")
    
    #    ob1_w, ob1_h, ob2_w, ob2_h = get_four_coords(gauss_distribution[i+1], gauss_distribution[i], gauss_distribution[i+2], gauss_distribution[i+3])
        ob1_w, ob1_h, ob2_w, ob2_h = get_fourcoords()
        
        random_val = random.random()
        
        if random_val > 0.7:
            objects_list.append(process_object(objects[random.randint(0, all_objects_count)], ob1_w, ob1_h))
            objects_list.append(process_object(objects[random.randint(0, all_objects_count)], ob2_w, ob2_h))
        else:
            objects_list.append(process_object(objects[random.randint(0, all_objects_count)], ob1_w, ob1_h))
            
            
            
        
        bg_w, bg_h = bg.size
    
        normal_image, normal_boxes, labels = create_new_image(objects_list)
        save_images(normal_image, normal_boxes, 'normal', file_count, labels)
        file_count+=1
        
        horizontal_flip_image, horizontal_flip_boxes = utils.HorizontalFlip()(normal_image.copy(), normal_boxes.copy())
        save_images(horizontal_flip_image, horizontal_flip_boxes,  'horizontal', file_count, labels)
          
#        for i in range(0, 3):
#            
#            normal_image_with_sun = add_sun_light(normal_image)
#            save_images(normal_image_with_sun, normal_boxes, 'normal_image_with_sun',file_count, labels)
#            file_count+=1
        
        for i in range(0, 1):
                
            normal_image, normal_boxes, labels = create_new_image(objects_list)
            scaled_image, scaled_boxes = utils.RandomScale((-.2, 0.2), diff = True)(normal_image.copy(), normal_boxes.copy())
            save_images(scaled_image, scaled_boxes, 'scale',file_count, labels)
            file_count+=1
            
        for i in range(0, 1):
               
            normal_image, normal_boxes, labels = create_new_image(objects_list)
            bright_cont_img = change_brightness(Image.fromarray(cv2.cvtColor(normal_image.copy(), cv2.COLOR_BGR2RGB)), random.uniform(.8,1.8))
            bright_cont_img = cv2.cvtColor(np.asarray(bright_cont_img), cv2.COLOR_RGB2BGR)
            save_images(bright_cont_img, normal_boxes, 'brightness', file_count, labels)
            file_count+=1
            
        for i in range(0, 1):
            noise_type = noises[random.randint(0, 1)]
            bg = Image.open(backgrounds[random.randint(0, bg_count-1)])
            
            if noise_type == "s&p":
                
                normal_image, normal_boxes, labels = create_new_image(objects_list)    
                normal_image = cv2.cvtColor(np.asarray(normal_image), cv2.COLOR_RGB2GRAY)
    #            resized_im = cv2.cvtColor(np.asarray(resized_im), cv2.COLOR_GRAY2BGRA)
                normal_image  = cv2.cvtColor(noisy(noise_type, normal_image).astype(np.uint8), cv2.COLOR_BGRA2RGBA)
                save_images(normal_image, normal_boxes, 'noise_' + noise_type, file_count, labels)
                
            else:

                resized_im, resize_w, resize_h, w_scale_factor, h_scale_factor, obj_class = objects_list[0]
                
                blur_img  = cv2.cvtColor(noisy(noise_type, cv2.cvtColor(np.asarray(resized_im), cv2.COLOR_RGBA2BGRA)).astype(np.uint8), cv2.COLOR_BGRA2RGBA)
                blur_bg  = cv2.cvtColor(noisy(noise_type, cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGBA2BGRA)).astype(np.uint8), cv2.COLOR_BGRA2RGBA)
                                
                normal_image, normal_boxes = create_new_blur_image(Image.fromarray(blur_img), w_scale_factor, h_scale_factor, obj_class, Image.fromarray(blur_bg))
    
                save_images(normal_image.astype(np.uint8), normal_boxes.copy(),  'noise_' + noise_type, file_count, [obj_class])
            
            file_count+=1
    
        for i in range(0, 1):
            
            normal_image, normal_boxes, labels = create_new_image(objects_list)
            
            shear_image, shear_boxes = utils.RandomShear(shear_factor = (-0.3, 0.3))(normal_image.copy(), normal_boxes.copy())
            save_images(shear_image, shear_boxes, 'shear', file_count, labels)
            file_count+=1
              
        for i in range(0, 1):
            
            normal_image, normal_boxes, labels = create_new_image(objects_list)
            
            hsv_image, hsv_boxes = utils.RandomHSV(40, 40, 30)(normal_image.copy(), normal_boxes.copy())
            save_images(hsv_image, hsv_boxes, 'hsv', file_count, labels)
            file_count+=1
   
        for i in range(0, 1):
            
            normal_image, normal_boxes, labels = create_new_image(objects_list)
            
            seq = utils.Sequence([utils.RandomHorizontalFlip(0.5), utils.RandomScale((0.1, 0.2), diff = True),
                                  utils.RandomShear(shear_factor = (-0.2, 0.2))])
            try:
                seq_image, seq_bboxes = seq(normal_image.copy(), normal_boxes.copy())
                save_images(seq_image, seq_bboxes, 'seq', file_count, labels)
                file_count+=1
            except:
                print("<<<<<<<<< Error in Augmentation >>>>>>>>>")
                pass
    
        
        count+=1
        file_count += 1
    
        if file_count % 2000 == 0:
            print('no.of images completed: {}'.format(count))
        
    
    # Augment data from the Hand annotated files.

    output_file.close()
          
