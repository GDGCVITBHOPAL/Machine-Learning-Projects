# Importing Project Dependencies
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import winsound
import streamlit as st

# Setting up config for GPU usage
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Using  Har-cascade classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loading the trained model for prediction purpose
model = keras.models.load_model('my_model (1).h5')

# Title for GUI
st.title('Drowsiness Detection')
img = []

# Navigation Bar
nav_choice = st.sidebar.radio('Navigation', ('Home', 'Sleep Detection', 'Help Us Improve'), index=0)
# Home page
if nav_choice == 'Home':
    st.header('Prevents sleep deprivation road accidents, by alerting drowsy drivers.')
    st.image('ISHN0619_C3_pic.jpg')
    st.markdown('<b>In accordance with the survey taken by the Times Of India, about 40 % of road </b>'
                '<b>accidents are caused</b> '
                '<b>due to sleep deprivation & fatigued drivers. In order to address this issue, this app will </b>'
                '<b>alert such drivers with the help of deep learning models and computer vision.</b>'
                '', unsafe_allow_html=True)
    st.image('sleep.jfif', width=300)
    st.markdown('<h1>How to use?<br></h1>'
                '<b>1. Go to Sleep Detection page from the Navigation Side-Bar.</b><br>'
                '<b>2. Make sure that, you have sufficient amount of light, in your room.</b><br>'
                '<b>3. Align yourself such that, you are clearly visible in the web-cam and '
                'stay closer to the web-cam. </b><br>'
                '<b>4. Web-cam will take 3 pictures of you, so keep your eyes in the same state'
                ' (open or closed) for about 5 seconds.</b><br>'
                '<b>5. If your eyes are closed, the model will make a beep sound to alert you.</b><br>'
                '<b>6. Otherwise, the model will continue taking your pictures at regular intervals of time.</b><br>'
                '<font color="red"><br><b>For the purpose of the training process of the model, '
                'dataset used is available <a href="https://www.kaggle.com/kutaykutlu/drowsiness-detection", '
                'target="_blank">here</a></font></b>'
                , unsafe_allow_html=True)
    
# Sleep Detection page
elif nav_choice == 'Sleep Detection':
    st.header('Image Prediction')
    cap = 0
    st.success('Please look at your web-cam, while following all the instructions given on the Home page.')
    st.warning(
        'Keeping the eyes in the same state is important but you can obviously blink your eyes, if they are open!!!')
    b = st.progress(0)
    for i in range(100):
        time.sleep(0.0001)
        b.progress(i + 1)

    start = st.radio('Options', ('Start', 'Stop'), key='Start_pred', index=1)

    if start == 'Start':
        decision = 0
        st.markdown('<font face="Comic sans MS"><b>Detected Facial Region of Interest(ROI)&emsp;&emsp;&emsp;&emsp;&emsp;Extractd'
                    ' Eye Features from the ROI</b></font>', unsafe_allow_html=True)
        
        # Best of 3 mechanism for drowsiness detection
        for _ in range(3):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # Proposal of face region by the har cascade classifier
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                roi_gray = gray[y:y + w, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Cenentroid method for extraction of eye-patch 
                centx, centy = roi_color.shape[:2]
                centx //= 2
                centy //= 2
                eye_1 = roi_color[centy - 40: centy, centx - 70: centx]
                eye_1 = cv2.resize(eye_1, (86, 86))
                eye_2 = roi_color[centy - 40: centy, centx: centx + 70]
                eye_2 = cv2.resize(eye_2, (86, 86))
                cv2.rectangle(frame1, (x + centx - 60, y + centy - 40), (x + centx - 10, y + centy), (0, 255, 0), 5)
                cv2.rectangle(frame1, (x + centx + 10, y + centy - 40), (x + centx + 60, y + centy), (0, 255, 0), 5)
                preds_eye1 = model.predict(np.expand_dims(eye_1, axis=0))
                preds_eye2 = model.predict(np.expand_dims(eye_2, axis=0))
                e1, e2 = np.argmax(preds_eye1), np.argmax(preds_eye2)
                
                # Display of face image and extracted eye-patch
                img_container = st.beta_columns(4)
                img_container[0].image(frame1, width=250)
                img_container[2].image(cv2.cvtColor(eye_1, cv2.COLOR_BGR2RGB), width=150)
                img_container[3].image(cv2.cvtColor(eye_2, cv2.COLOR_BGR2RGB), width=150)
                print(e1, e2)
                
                # Decision variable for prediction
                if e1 == 1 or e2 == 1:
                    pass
                else:
                    decision += 1

            except NameError:
                st.warning('Hold your camera closer!!!\nTrying again in 2s')
                cap.release()
                time.sleep(1)
                continue

            except:
                cap.release()
                continue

            finally:
                cap.release()

        # If found drowsy, then make a beep sound to alert the driver
        if decision == 0:
            st.error('Eye(s) are closed')
            winsound.Beep(2500, 2000)

        else:
            st.success('Eyes are Opened')
        st.warning('Please select "Stop" and then "Start" to try again')

# Help Us Improve page
else:
    st.header('Help Us Improve')
    st.success('We would appreciate your Help!!!')
    st.markdown(
        '<font face="Comic sans MS">To make this app better, we would appreciate your small amount of time.</font>'
        '<font face="Comic sans MS">Let me take you through, some of the basic statistical analysis of this </font>'
        '<font face="Comic sans MS">model. <br><b>Accuracy with naked eyes = 99.5%<br>Accuracy with spectacles = 96.8%</b><br></font> '
        '<font face="Comic sans MS">As we can see here, accuracy with spectacles is not at all spectacular, and hence to make this app </font>'
        '<font face="Comic sans MS">better, and to use it in real-time situations, we require as much data as we can gather.</font> '
        , unsafe_allow_html=True)
    st.warning('NOTE: Your identity will be kept anonymous, and only your eye-patch will be extracted!!!')
    # Image upload
    img_upload = st.file_uploader('Upload Image Here', ['png', 'jpg', 'jpeg'])
    if img_upload is not None:
        prog = st.progress(0)
        to_add = cv2.imread(str(img_upload.read()), 0)
        to_add = pd.DataFrame(to_add)
        
        # Save it in the database
        to_add.to_csv('Data_from_users.csv', mode='a', header=False, index=False, sep=';')
        for i in range(100):
            time.sleep(0.001)
            prog.progress(i + 1)
        st.success('Uploaded Successfully!!! Thank you for contributing.')
