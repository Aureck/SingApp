import os, cv2
from datetime import datetime
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, there_hand
from constants import FRAME_ACTIONS_PATH, FONT, FONT_SIZE, FONT_POS, MODEL_FRAMES, MIN_LENGTH_FRAMES

def save_frames(frames, out_dir):
    for i, f in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(out_dir, f"{i:02}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])

def capture_samples(word_name: str, margin_frame=1, min_frames=5, delay_frames=3):
    word_dir = os.path.join(FRAME_ACTIONS_PATH, word_name)
    create_folder(word_dir)

    count, fix, recording = 0, 0, False
    frames = []

    with Holistic() as holistic:
        cam = cv2.VideoCapture(0)
        while cam.isOpened():
            ok, frame = cam.read()
            if not ok: break

            image = frame.copy()
            results = mediapipe_detection(frame, holistic)

            if there_hand(results) or recording:
                recording = False
                count += 1
                if count > margin_frame:
                    cv2.putText(image, "Capturando...", FONT_POS, FONT, FONT_SIZE, (255,50,0), 2)
                    frames.append(frame.copy())
            else:
                if len(frames) >= (min_frames + margin_frame):
                    fix += 1
                    if fix < delay_frames:
                        recording = True
                    else:
                        frames = frames[:-(margin_frame + delay_frames)]  
                        sample_id = datetime.now().strftime("sample_%y%m%d%H%M%S%f")
                        out_dir = os.path.join(word_dir, sample_id)
                        create_folder(out_dir)
                        save_frames(frames, out_dir)
                        print(f"[OK] {word_name}: {len(frames)} frames -> {out_dir}")
                # reset
                recording, fix = False, 0
                frames, count = [], 0
                cv2.putText(image, "Listo para capturar...", FONT_POS, FONT, FONT_SIZE, (0,220,100), 2)

            draw_keypoints(image, results)
            cv2.imshow(f'Captura "{word_name}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_samples("hola")
