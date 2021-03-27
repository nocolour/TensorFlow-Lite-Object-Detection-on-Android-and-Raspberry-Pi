import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="TFLite_model/detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('TFLite_model/labelmap.txt', 'r') as F:
    class_names = F.readlines()

WINNAME = "Capture"
FRAME_INTERVAL = 30  # msec

CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600
CENTER_CROP_X1 = int((CAPTURE_WIDTH - CAPTURE_HEIGHT) / 2)
CENTER_CROP_X2 = CAPTURE_WIDTH - CENTER_CROP_X1
DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 300
colors = ((255, 255, 0), (0, 255, 255), (128, 256, 128), (64, 192, 255), (128, 128, 255)) * 2

ret = False
while ret == False:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    ret, img = cap.read()

key = 0
while key != ord('q'):
    ret, img = cap.read()
    # img = img[:, CENTER_CROP_X1:CENTER_CROP_X2]  # crop center square
    img = cv2.flip(img, 1)
    x = cv2.resize(img, (300, 300))  # input size of coco ssd mobilenet?
    x = x[:, :, [2, 1, 0]]  # BGR -> RGB
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    tflite_results1 = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
    tflite_results2 = interpreter.get_tensor(output_details[1]['index'])  # Classes (0=Person)
    tflite_results3 = interpreter.get_tensor(output_details[2]['index'])  # Scores
    tflite_results4 = interpreter.get_tensor(output_details[3]['index'])  # Number of detections

    img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    for i in range(int(tflite_results4[0])):
        (top, left, bottom, right) = tflite_results1[0, i] * 300
        class_name = class_names[tflite_results2[0, i].astype(int) + 1].rstrip()
        prob = tflite_results3[0, i]
        if prob >= 0.5:
            print("Location=({},{})-({},{})".format(int(left), int(top), int(right), int(bottom)))
            print("Class={}".format(class_name))
            print("Probability={}".format(prob))
            left = int(left * DISPLAY_WIDTH / 300)
            right = int(right * DISPLAY_WIDTH / 300)
            top = int(top * DISPLAY_HEIGHT / 300)
            bottom = int(bottom * DISPLAY_HEIGHT / 300)
            cv2.rectangle(img, (left, top), (right, bottom), colors[i], 1)
            cv2.rectangle(img, (left, top + 20), (left + 160, top), colors[i], cv2.FILLED)
            cv2.putText(img, "{} ({:.3f})".format(class_name, prob),
                        (left, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow(WINNAME, img)
    cv2.moveWindow(WINNAME, 0, 0)

    key = cv2.waitKey(FRAME_INTERVAL)
    if key == ord('s'):
        cv2.imwrite('result.jpg', img)

cap.release()
cv2.destroyAllWindows()
