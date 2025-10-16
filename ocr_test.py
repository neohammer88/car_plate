import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img = cv2.imread("plate_sample.png")
if img is None:
    print("이미지 로드 실패")
    exit()

results = ocr.predict(img)

for res in results:
    print("\n==== OCR 결과 ====")
    try:
        # 이미 dict라면 함수 호출 없이 사용
        if isinstance(res, dict):
            if "res" in res and "output" in res["res"]:
                for item in res["res"]["output"]:
                    print("텍스트:", item.get("text", "없음"), "| 정확도:", item.get("score", 0))
            else:
                print("output 없음:", res)
        else:
            print("예상치 못한 타입:", type(res))
    except Exception as e:
        print("파싱 중 예외 발생:", e)
