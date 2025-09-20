import cv2, argparse, os
import numpy as np
from pipeline import Pipeline

def make_synthetic():
    img = cv2.imread("../synthetic_demo.png")
    return img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="Path to image")
    ap.add_argument("--video", type=str, default=None, help="Path to video file")
    ap.add_argument("--source", type=str, default=None, choices=["synthetic"], help="Use synthetic demo frame")
    ap.add_argument("--save", action="store_true", help="Save annotated frames to out.mp4/out.png instead of imshow() for headless use" )
    args = ap.parse_args()

    pipe = Pipeline()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    if args.image or args.source == "synthetic":
        frame = make_synthetic() if args.source == "synthetic" else cv2.imread(args.image)
        dets, r, rs = pipe.step(frame)
        out = pipe.annotate(frame.copy(), dets, r, rs)
        if args.save:
            cv2.imwrite("out.png", out); print("Saved out.png")
        else:
            cv2.imshow("out", out); cv2.waitKey(0)
        pipe.close()
        raise SystemExit(0)

    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if args.save:
        ret, test = cap.read()
        if not ret: raise SystemExit("Failed to read source")
        h,w = test.shape[:2]
        writer = cv2.VideoWriter("out.mp4", fourcc, 20.0, (w,h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        dets, r, rs = pipe.step(frame)
        out = pipe.annotate(frame.copy(), dets, r, rs)
        if args.save:
            writer.write(out)
        else:
            cv2.imshow("situational_analysis", out)
            if cv2.waitKey(1) & 0xFF == 27: break

    if writer: writer.release()
    cap.release()
    pipe.close()
    cv2.destroyAllWindows()
