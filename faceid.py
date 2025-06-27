# import kivy dependencies 
import cv2.data
from kivy.app import App 
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components 
from kivy.uix.image import Image 
from kivy.uix.button import Button 
from kivy.uix.label import Label

# Import other kivy stuff 
from kivy.clock import Clock
from kivy.graphics.texture import Texture 
from kivy.logger import Logger 

# Import other dependencies 
import cv2 
import tensorflow as tf 
from layers import L1Dist
import os 
import numpy as np 

# Build app and layout 

class CamApp(App): 

    def build(self): 
        self.img1 = Image(size_hint = (1, .8))
        self.button = Button(text = 'Verify',on_press = self.verify, size_hint = (1, .1))
        self.verification_label = Label(text = 'Verification Uninitiated', size_hint = (1, .1))

        # add items to layout 
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tf/keras model 
        self.model = tf.keras.models.load_model('siamese_model.keras', custom_objects = {'L1Dist': L1Dist})

        #Setup video capture device 
        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    # Run continuously to get webcam feed
    def update(self, *args): 
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            for (x, y, w, h) in faces: 
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop and resize face
                face_crop = frame[y:y+h, x:x+w]
                self.resized_face = cv2.resize(face_crop, (47, 62))

                # Flip and convert to texture
                buf = cv2.flip(self.resized_face, 0).tobytes()
                img_texture = Texture.create(size=(self.resized_face.shape[1], self.resized_face.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.img1.texture = img_texture
                break  # Only show one face at a time
    
    # Load image from file and convert to 100, 100 pixels 
    def preprocess(self,file_path): 
        # Read in image from file path 
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing 
        img = tf.image.resize(img, (100,100))

        # Scale image to be between 0 and 1 
        img = img/255.0

        return img 
    # verification function to verify
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5
        results = []

        # Save the input face image
        if hasattr(self, 'resized_face'):
            SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            cv2.imwrite(SAVE_PATH, self.resized_face)
            self.verification_label.text = 'Image saved successfully'
        else:
            self.verification_label.text = 'No face to save yet.'
            return

        # Preprocess input image
        input_img = self.preprocess(SAVE_PATH)
        verification_dir = os.path.join('application_data', 'verification_images')

        for image in os.listdir(verification_dir):
            validation_img = self.preprocess(os.path.join(verification_dir, image))

            # Predict similarity
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result[0][0])  # Extract scalar value

        # Verification logic
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(results)
        verified = verification > verification_threshold

        if verified:
            self.verification_label.text = 'Verified'
            self.button.background_color = (0, 1, 0, 1)  # Green
        else:
            self.verification_label.text = 'Unverified'
            self.button.background_color = (1, 0, 0, 1)  # Red


        # Logs
        Logger.info(f"All results: {results}")
        Logger.info(f"Results > 0.5: {np.sum(np.array(results) > 0.9)}")
        Logger.info(f"Results > 0.5: {np.sum(np.array(results) > 0.5)}")
        Logger.info(f"Results > 0.2: {np.sum(np.array(results) > 0.2)}")
        Logger.info(f"Results > 0.3: {np.sum(np.array(results) > 0.3)}")
        Logger.info(f"Results > 0.4: {np.sum(np.array(results) > 0.4)}")
        Logger.info(f"Verified: {verified} ({detection} passed out of {len(results)})")

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        

        return results, verified





if __name__ == '__main__': 
    CamApp().run()
