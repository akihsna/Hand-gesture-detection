import cv2
import mediapipe as mp

# Setup
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame not captured")
            break

        image = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture = "No hand detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark

                fingers = []

                fingers.append(1 if lm[4].x < lm[3].x else 0)
                fingers += [1 if lm[i].y < lm[i - 2].y else 0 for i in [8, 12, 16, 20]]

                if fingers == [0, 0, 0, 0, 0]:
                    gesture = "A"
                elif fingers == [0, 1, 1, 1, 1]:
                    gesture = "B"
                elif fingers == [0, 1, 1, 1, 0]:
                    gesture = "C"
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = "L"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = "V"
                else:
                    gesture = "Unknown"

        cv2.putText(image, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Sign Language Detector", image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:  # ESC key bhi chalega
            print("👋 Quit signal received")
            break

except KeyboardInterrupt:
    print("⚠️ Interrupted by keyboard")

finally:
    print("🔒 Releasing camera and closing windows")
    cap.release()
    cv2.destroyAllWindows()
