import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model/best_model.h5')
gestures = {0:'Batu', 1:'Kertas', 2:'Gunting'}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    pred = model.predict(img[np.newaxis,...], verbose=0)
    label = gestures[np.argmax(pred)]
    cv2.putText(frame, label, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
    cv2.imshow('Batu Kertas Gunting - 99%+ Akurat', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()