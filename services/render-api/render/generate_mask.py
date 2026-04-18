"""
Generate a binary person mask video from a SOMA render.
Background is pure black (0,0,0) — anything above threshold is person.
Output: white person, black background (grayscale mp4).
"""
import sys, cv2, numpy as np

def generate_mask(input_path: str, output_path: str, threshold: int = 10):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Any pixel brighter than threshold is person
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        mask[gray > threshold] = 255
        # Dilate slightly to cover edges
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        out.write(mask)
    
    cap.release()
    out.release()
    print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    generate_mask(sys.argv[1], sys.argv[2])
