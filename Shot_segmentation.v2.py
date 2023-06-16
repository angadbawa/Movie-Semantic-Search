from skimage.metrics import structural_similarity
#import argparse
import imutils
import os
import cv2

i=0
j=1
shotcount=1
frame_width = 1280
frame_height = 544

 #/home/u19581/Movie Semantic Search/Movies

# Directory for movie shots 
directory = r'/home/u19581/Movie Semantic Search/Movies/movie_shots'

os.chdir(directory)

# create videowriter object 
out_mp4 = cv2.VideoWriter('shot1.mp4',cv2.VideoWriter_fourcc(*'XVID'), 24, (frame_width,frame_height))

while i<=10000:
    
    # changing to shots directory 
    os.chdir(directory)

    # reading image as an array 
    imageA = cv2.imread("/home/u19581/Movie Semantic Search/Movies/Movie_frames_2/Frame"+str(i)+".png")
    imageB = cv2.imread("/home/u19581/Movie Semantic Search/Movies/Movie_frames_2/Frame"+str(i+1)+".png")
    
    # Converting to grey scale for ssim
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # converting to hsv for histogram 
    hsvA = cv2.cvtColor(imageA,cv2.COLOR_BGR2HSV)
    hsvB = cv2.cvtColor(imageB,cv2.COLOR_BGR2HSV)

    #hist0= cv2.calcHist([imageA], [0, 1, 2], None, [70, 70, 70],[0, 256, 0, 256, 0, 256])
    #hist0=cv2.normalize(hist0, hist0).flatten()

    # calculating histogram score(img,channels,mask,binsize,range of pixelvalues)
    hist1 = cv2.calcHist([hsvA], [0, 1, 2], None, [70, 70, 70],[0, 180, 0, 265, 0, 256])
    hist1 = cv2.normalize(hist1, hist1,0, 1, cv2.NORM_MINMAX)
    h1 = hist1

    hist = cv2.calcHist([hsvB], [0, 1, 2], None, [70, 70, 70],[0, 180, 0, 265, 0, 256])
    hist = cv2.normalize(hist, hist,0, 1, cv2.NORM_MINMAX)
    h2 = hist
    
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    
    diff = (diff * 255).astype("uint8")
    
    # print("{}          {}         {}".format(i,d,score))
    
    #hsv_x.append(i+1)
    #hsv_y.append(d)
    
    
    if d<0.4 or score >=0.75:
        # if frame is similar then include in a shot
        out_mp4.write(imageA)
        
    else:
        # when not similar break the shot and add prev frame to last shot
        out_mp4.write(imageA)

        # relase the shot and incr the counter 
        out_mp4.release()
        shotcount+=1

        # create new videowriter obj for next shot 
        out_mp4 = cv2.VideoWriter('shot'+str(shotcount)+'.mp4',cv2.VideoWriter_fourcc(*'XVID'), 24, (frame_width,frame_height))
        
    i+=1
    
    
# once finished release object 
cv2.destroyAllWindows()