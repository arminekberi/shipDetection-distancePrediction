import cv2

MAX_INDEX_TO_TRY = 5


def main():
    print('Opening each camera index in turn. Press any key to go to the next one, or "q" to stop.')
    for index in range(MAX_INDEX_TO_TRY):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        h, w = frame.shape[:2]
        print(f'index {index}: {w}x{h}')
        cv2.putText(frame, f'index={index}  ({w}x{h})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('camera index probe - press any key for next, q to quit', frame)
        key = cv2.waitKey(0) & 0xFF
        cap.release()
        cv2.destroyAllWindows()
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
