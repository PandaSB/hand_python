import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands=mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

try:
    webcam = cv2.VideoCapture(2)
    while webcam.isOpened():
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        success, img = webcam.read();
        if success:
            img=cv2.flip(img,1)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = img.shape

            facemeshresults = mp_face_mesh.process(img)
            handresults = mp_hands.process(img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            recognition_result = recognizer.recognize(mp_image)

            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            if handresults.multi_hand_landmarks:
                for hand_landmarks in handresults.multi_hand_landmarks:
                   mp_drawing.draw_landmarks(img,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS,mp.solutions.drawing_styles.get_default_hand_landmarks_style(),mp.solutions.drawing_styles.get_default_hand_connections_style())

            if facemeshresults.multi_face_landmarks:
                for face_landmarks in facemeshresults.multi_face_landmarks:
                    mp_drawing.draw_landmarks(img,face_landmarks,mp.solutions.face_mesh.FACEMESH_IRISES,None,mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        
            if facemeshresults.multi_face_landmarks:
                for face_landmarks in facemeshresults.multi_face_landmarks:
                    mp_drawing.draw_landmarks(img,face_landmarks,mp.solutions.face_mesh.FACEMESH_CONTOURS,None,mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                gesture_text = f"Gesture : Top {top_gesture.category_name}: {top_gesture.score:.2f} "
                for gesture in recognition_result.gestures:
                    for category in gesture:
                        gesture_text += f" - {category.category_name}: {category.score:.2f}"



                if gesture_text:
                    text_position = (10, 50) # Bottom-left corner of the text string in the image (x, y)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 255, 0) # BGR color (Green)
                    thickness = 2
                    line_type = cv2.LINE_AA

                    cv2.putText(img, gesture_text,text_position, font, font_scale, font_color, thickness, line_type)    


       
            cv2.imshow("video",img)
        else:
            print("Ignore bad frame")
            continue
except Exception as e:
    print(e)    
webcam.release()
cv2.destroyAllWindows()