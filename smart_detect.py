from detect import analyze_image as yolo_analyze
from vlm import analyze_with_vlm
from PIL import Image

def smart_analyze(image: Image.Image) -> dict:
    # YOLO 기반 감지
    yolo_result = yolo_analyze(image)

    # VLM 기반 설명 추가
    vlm_caption = analyze_with_vlm(image)

    # 통합 요약 생성
    full_summary = f"{yolo_result['summary']}\nVLM says: {vlm_caption}"

    # 기존 위험 정보 유지 + 설명 추가
    return {
        "summary": full_summary,
        "labels": yolo_result["labels"],
        "signal_color": yolo_result["signal_color"],
        "hazard_level": yolo_result["hazard_level"],
        "direction": yolo_result["direction"]
    }
