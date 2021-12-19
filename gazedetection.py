

import cv2
import dlib
from scipy.spatial import distance

def aspect_ratio(data):
  v1 = distance.euclidean(data[1], data[5])
  v2 = distance.euclidean(data[2], data[4])
  h = distance.euclidean(data[0], data[3])
  aspectRatio = (v1+v2)/(2*h)
  return aspectRatio

capture = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while True:
      _, frame = capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      faces = hog_face_detector(gray)
      for face in faces:
          points = dlib_facelandmark(gray, face)
          left_points = []
          right_points = []
          
          for n in range(36, 42):
              x_cords = points.part(n).x
              y_cords = points.part(n).y
              left_points.append((x_cords, y_cords))
              '''next = n+1
              if(n==41):
                  next = 36
              x2_cords = points.part(next).x
              y2_cords = points.part(next).y
              cv2.line(frame,(x_cords,y_cords),(x2_cords,y2_cords),(0,255,0),1)'''
        
          for n in range(42, 48):
              x_cords = points.part(n).x
              y_cords = points.part(n).y
              right_points.append((x_cords, y_cords))
              '''next = n+1
              if(n==47):
                  next = 42
              x2_cords = points.part(next).x
              y2_cords = points.part(next).y
              cv2.line(frame,(x_cords,y_cords),(x2_cords,y2_cords),(0,255,0),1)'''

          left_ar = aspect_ratio(left_points)
          right_ar = aspect_ratio(right_points)
          ear = (left_ar+right_ar)/2
          
          left_ar = round(left_ar, 2)
          right_ar = round(right_ar, 2)
          ear = round(ear, 2)
          
          if(left_ar<right_ar and left_ar<0.22):
              cv2.putText(frame, "Right Eye", (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255),4)
          elif(right_ar<left_ar and right_ar<0.22):
              cv2.putText(frame, "Leftt Eye", (20, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255),4)
 

      cv2.imshow("Video feed on", frame)
      key = cv2.waitKey(1)
      if key == 27:
          break



capture.release()
cv2.destroyAllWindows()