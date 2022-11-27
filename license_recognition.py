import cv2 as cv



def license_plate(video):
    license_plate = cv.CascadeClassifier('files/haarcascade_russian_plate_number.xml')
    captured = cv.VideoCapture(video)
    count = 0
    while True:
        success , frame = captured.read()
        convert_to_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        all_licenses = license_plate.detectMultiScale(convert_to_gray,scaleFactor=2, minNeighbors=3 )
        for (x,y,w,h) in all_licenses:
            image_area = w * h
            license_img = frame[y: y+h ,x: x+w]
            color = (0,255,255)
            thickness = 2
            cord_x = x + w
            cord_y = y + h
            cv.rectangle(frame,(x,y),(cord_x,cord_y),color, thickness)
        cv.imshow('frame',frame)
        if (cv.waitKey(1) & 0xFF) == ord('s'):
            cv.imwrite(f'snapshots/{count}.jpg', license_img)
            count += 1    
    
    
license_plate('files/video12.mp4')    
    