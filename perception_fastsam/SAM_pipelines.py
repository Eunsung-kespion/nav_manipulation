"""
FastSAM Pipeline Utilities
Utility functions for FastSAM-based segmentation
"""
import os
import logging
import numpy as np
from PIL import Image
import cv2
import base64
import io
from fastsam import FastSAM, FastSAMPrompt

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """PIL Image를 Base64 문자열로 변환"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def find_robust_center(mask, image_width, image_height):
    """
    거리 변환(Distance Transform)을 사용하여 마스크의 가장 안쪽 중심점을 찾습니다.

    Args:
        mask (np.ndarray): 부울(boolean) 또는 0/1 형태의 마스크 배열.
        image_width (int): 원본 이미지의 너비.
        image_height (int): 원본 이미지의 높이.

    Returns:
        tuple or None: ((center_y, center_x), normalized_center) 또는 실패 시 (None, None).
    """
    # 마스크가 비어있는지 확인
    if np.count_nonzero(mask) == 0:
        logger.warning("Mask is empty, cannot find a robust center.")
        return None, None

    # cv2.distanceTransform은 uint8 타입의 입력을 요구합니다.
    mask_uint8 = mask.astype(np.uint8)
    
    # 거리 변환 수행. DIST_L2는 유클리드 거리를 의미합니다.
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    
    # 거리 값이 최대인 지점의 좌표를 찾습니다.
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    # max_loc은 (x, y) 순서의 튜플입니다.
    center_x, center_y = max_loc
    
    # 정규화된 좌표 계산 (Python native types로 변환)
    normalized_center = (float(center_x / image_width), float(center_y / image_height))
    
    logger.info(f"Found robust center via Distance Transform at: ({center_x}, {center_y})")
    
    return (int(center_y), int(center_x)), normalized_center


def fastsam_text_segmentation(
    model,
    image,
    text_prompt,
    output_dir,
    device="cuda",
    imgsz=1024,
    conf=0.4,
    iou=0.9,
    retina=True
):
    """
    FastSAM을 사용한 텍스트 프롬프트 기반 세그멘테이션
    
    Args:
        model: FastSAM 모델 인스턴스
        image: PIL Image 또는 이미지 경로
        text_prompt: 텍스트 프롬프트 (예: "a dog", "red brick")
        output_dir: 출력 디렉토리
        device: 디바이스 ("cuda" 또는 "cpu")
        imgsz: 이미지 크기
        conf: confidence threshold
        iou: IOU threshold
        retina: retina masks 사용 여부
    
    Returns:
        dict: 세그멘테이션 결과 정보
    """
    logger.info(f"Starting FastSAM segmentation for text: '{text_prompt}'")
    
    try:
        # 이미지 로드
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError("Image must be a file path or PIL Image")
        
        # FastSAM 추론 실행
        logger.info("Running FastSAM inference...")
        everything_results = model(
            pil_image,
            device=device,
            retina_masks=retina,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )
        
        # FastSAMPrompt를 사용하여 text prompt 적용
        prompt_process = FastSAMPrompt(pil_image, everything_results, device=device)
        
        # Text prompt로 segmentation 수행
        logger.info(f"Applying text prompt: '{text_prompt}'")
        ann = prompt_process.text_prompt(text=text_prompt)
        
        # 결과 처리
        if ann is None or len(ann) == 0:
            logger.warning(f"No segmentation result for text: '{text_prompt}'")
            return {
                "original": str(image) if isinstance(image, str) else "PIL_Image",
                "mask_image_base64": None,
                "mask_path": None,
                "normalized_obj_center": None,
                "message": f"No object found for '{text_prompt}'"
            }
        
        # 마스크 추출
        if hasattr(ann, 'masks') and ann.masks is not None:
            mask = ann.masks[0].cpu().numpy()  # 첫 번째 마스크 선택
        else:
            logger.warning("No masks found in annotations")
            return {
                "original": str(image) if isinstance(image, str) else "PIL_Image",
                "mask_image_base64": None,
                "mask_path": None,
                "normalized_obj_center": None,
                "message": "No masks found in result"
            }
        
        # 마스크를 0-255 범위로 변환
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # PIL Image로 변환
        mask_image = Image.fromarray(mask_uint8)
        
        # Base64 인코딩
        mask_image_base64 = image_to_base64(mask_image)
        
        # 파일로 저장
        os.makedirs(output_dir, exist_ok=True)
        safe_text_prompt = "".join(c for c in text_prompt if c.isalnum() or c in " _-").rstrip()
        mask_filename = f"mask_{safe_text_prompt}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        mask_image.save(mask_path)
        logger.info(f"Saved mask to: {mask_path}")
        
        # 마스크의 중심점 계산
        (center_y, center_x), normalized_center = find_robust_center(
            mask, pil_image.width, pil_image.height
        )
        
        if normalized_center is None:
            logger.warning("Could not find robust center for mask")
        
        return {
            "original": str(image) if isinstance(image, str) else "PIL_Image",
            "mask_image_base64": mask_image_base64,
            "mask_path": mask_path,
            "normalized_obj_center": normalized_center,
            "message": "Segmentation complete"
        }
        
    except Exception as e:
        logger.error(f"Error in FastSAM segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("FastSAM Text Prompt Segmentation", add_help=True)
    parser.add_argument("--image_path", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help='text prompt eg: "a dog"')
    parser.add_argument("--model_path", type=str, default="./weights/FastSAM.pt", help="FastSAM model path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="directory to save results")
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--conf", type=float, default=0.4, help="object confidence threshold")
    parser.add_argument("--iou", type=float, default=0.9, help="iou threshold for filtering the annotations")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--retina", type=bool, default=True, help="draw high-resolution segmentation masks")
    
    args = parser.parse_args()
    
    # 모델 로드
    logger.info(f"Loading FastSAM model from: {args.model_path}")
    model = FastSAM(args.model_path)
    
    # 세그멘테이션 실행
    results = fastsam_text_segmentation(
        model=model,
        image=args.image_path,
        text_prompt=args.text_prompt,
        output_dir=args.output_dir,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina=args.retina
    )
    
    logger.info("Segmentation complete!")
    logger.info(f"Results: {results}")
