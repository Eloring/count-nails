import  cv2
import numpy as np
import hsvdict
import projection
from scipy.ndimage import measurements

def get_color(frame, tar_color):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    color_dict = hsvdict.getColorList()
    if tar_color in color_dict:
        mask = cv2.inRange(hsv,color_dict[tar_color][0],color_dict[tar_color][1])
        # cv2.imwrite(tar_color+'.jpg',mask)
        # binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # binary = cv2.dilate(binary,None,iterations=2)
        # img, cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # sum = 0
        # for c in cnts:
        #     sum+=cv2.contourArea(c)
        # if sum > maxsum :
        #     maxsum = sum
        #     color = d
        #     maxmask = mask
    return mask
 
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def box_POLYX(img,color):
    img = cv2.resize(img, (700,500))
    bimg = get_color(img, color)
    cv2.imwrite('temp/bimg.jpg', bimg)
    kernel = np.ones((3,3), np.uint8) 
    closing = cv2.morphologyEx(bimg, cv2.MORPH_CLOSE, kernel, iterations = 8)
    cv2.imwrite('temp/closing.jpg', closing)
    erosion = cv2.erode(closing, kernel, iterations = 1)
    cv2.imwrite('temp/erosion.jpg', erosion)
    # cv2.imwrite('result.jpg', erosion)
    labels, nbr_objects = measurements.label(erosion)

    dilate = cv2.dilate(erosion, kernel, iterations = 5)
    img_mix = cv2.bitwise_not(img,img, mask=dilate)
    # img_mix = cv2.addWeighted(img,0.5,bimg,0.5,0)
    img_path = 'temp/temp'+color+'.jpg'
    cv2.imwrite(img_path, img_mix)
    return str(nbr_objects),img_path

if __name__ == '__main__':

    # img1 = cv2.imread('purple.jpg')
    # img2 = cv2.imread('red.jpg')
    # cv2.imwrite('plusimage.jpg', img1+img2)

    img = cv2.imread('18100801.jpg')
    # resize
    while img.shape[0]>1000 and img.shape[1]>1000:
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    
    # eqimg = hisEqulColor(img)
    bimg = get_color(img, 'yellow')

    kernel = np.ones((3,3), np.uint8) 

    # opening = cv2.morphologyEx(bimg, cv2.MORPH_OPEN, kernel, iterations = 3)

    closing = cv2.morphologyEx(bimg, cv2.MORPH_CLOSE, kernel, iterations = 4)

    erosion = cv2.erode(closing, kernel, iterations = 2)

    cv2.imwrite('result.jpg', erosion)

    labels, nbr_objects = measurements.label(erosion)
    print("nbr_objects", nbr_objects)

    
