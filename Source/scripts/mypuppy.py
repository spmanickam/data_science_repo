import cv2
puppy_image = cv2.imread('../data/raw/Computer-Vision-with-Python_Joseph_Portilla/DATA/00-puppy.jpg')

while True:
    print('looping...press esc key to exit')
    cv2.imshow('my_puppy', puppy_image)
    # wait for 1 milli second and if the escape key is pressed, break out of the loop; can also use if cv2.waitKey(1) & 0xFF = ord('q'):
    if cv2.waitKey(1) & 0xFF == 27: 
        print('about to break')
        break
print('got out of the while loop')
cv2.destroyAllWindows()
exit(1)