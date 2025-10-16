import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img = cv2.imread("plate_sample.png")
if img is None:
    print("Fail to load a image")
    exit()

results = ocr.predict(img)

for res in results:
    print("\n==== OCR result ====")
    try:
        # no function if already
        if isinstance(res, dict):
            if "res" in res and "output" in res["res"]:
                for item in res["res"]["output"]:
                    print("Text:", item.get("text", "Nothing"), "| Trust:", item.get("score", 0))
            else:
                print("output Nothing:", res)
        else:
            print("Unexpected type:", type(res))
    except Exception as e:
        print("Exception:", e)
