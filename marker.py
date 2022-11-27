import cv2 as cv
import numpy as np









cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mask_color = [(124,123,123),(179,195,255)]
marker_color = [255,182,193]

my_coords = []


def colorfinder(frame, mask_color):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array(mask_color[0])
    higher = np.array(mask_color[1])
    mask_mask = cv.inRange(frame_hsv, lower, higher)
    x,y =getContours(mask_mask)
    cv.circle(frame, (x,y), 10, marker_color, cv.FILLED)    
    return [x,y]
def getContours(frame):
    contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for c in contours:
       area = cv.contourArea(c)
       if area > 500:
          parameter = cv.arcLength(c,True)
          approx=cv.approxPolyDP(c, 0.02*parameter, True)
          x,y,w,h = cv.boundingRect(approx)  
    return x+(w//2) , y

def coords_chaser(coords):
    for coord in coords:
        cv.circle(frame, (coord[0],coord[1]),10, marker_color, cv.FILLED)


while True:
    success , frame  = cap.read()
    new_coords = colorfinder(frame, mask_color)
    if new_coords != None :
        my_coords.append(new_coords)
    if len(my_coords):
        coords_chaser(my_coords)
                
    cv.imshow("frame", frame)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

