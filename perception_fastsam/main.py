"""
FastSAM Real-time Evaluation Server
WebSocket으로 받은 color 이미지를 사용하여 실시간 text prompt segmentation 수행
"""
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import cv2
import os
import uuid
from PIL import Image
import torch
import logging
import io
from typing import Set

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastSAM 관련 임포트
from fastsam import FastSAM, FastSAMPrompt
import clip


class OptimizedFastSAMPrompt(FastSAMPrompt):
    """CLIP 모델을 재사용하는 최적화된 FastSAMPrompt"""
    
    def __init__(self, image, results, device='cuda', clip_model=None, clip_preprocess=None):
        super().__init__(image, results, device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
    
    def text_prompt(self, text):
        """CLIP 모델을 재사용하는 text_prompt 오버라이드"""
        if self.results == None:
            return []
        
        format_results = self._format_results(self.results[0], 0)
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
        
        # CLIP 모델이 전달되었으면 재사용, 없으면 기존 방식 사용
        if self.clip_model is not None and self.clip_preprocess is not None:
            scores = self.retrieve(self.clip_model, self.clip_preprocess, cropped_boxes, text, device=self.device)
        else:
            # Fallback: 기존 방식 (CLIP 모델 로드)
            clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
            scores = self.retrieve(clip_model, preprocess, cropped_boxes, text, device=self.device)
        
        # CLIP 점수를 numpy array로 변환 (torch tensor일 수 있음)
        if hasattr(scores, 'cpu'):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)
        
        # 점수 통계 로깅
        max_score = float(scores_np.max())
        min_score = float(scores_np.min())
        mean_score = float(scores_np.mean())
        num_candidates = len(scores_np)
        
        logger.info(f"[CLIP] Text prompt: '{text}'")
        logger.info(f"[CLIP] Number of candidate masks: {num_candidates}")
        logger.info(f"[CLIP] Score statistics - Max: {max_score:.4f}, Min: {min_score:.4f}, Mean: {mean_score:.4f}")
        
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        selected_idx = max_idx + sum(np.array(filter_id) <= int(max_idx))
        selected_score = float(scores_np[max_idx])
        
        logger.info(f"[CLIP] Selected mask index: {selected_idx} (original: {max_idx}), Score: {selected_score:.4f}")
        
        # CLIP threshold 체크 (0.5)
        clip_threshold = 0.5
        if selected_score < clip_threshold:
            logger.warning(f"[CLIP] Selected score {selected_score:.4f} is below threshold {clip_threshold}, returning None")
            return None
        
        return np.array([annotations[selected_idx]['segmentation']])


class FastSAMEvaluator:
    """실시간 FastSAM Evaluation을 위한 클래스"""
    
    def __init__(self, config):
        self.config = config
        # MPS is only available on macOS, so check if the attribute exists
        mps_available = (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
        
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if mps_available
            else "cpu"
        )
        self.model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.is_initialized = False
        
        logger.info(f"[INIT] Device: {self.device}")
        
        # 출력 디렉토리 생성
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 모델 초기화
        self._load_models()
        
    def _load_models(self):
        """FastSAM 및 CLIP 모델 로드"""
        try:
            logger.info("Loading FastSAM model...")
            
            model_path = self.config['fastsam_model_path']
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FastSAM model not found at {model_path}")

            # FastSAM 모델 로드
            self.model = FastSAM(model_path)
            
            # CLIP 모델 로드 (한 번만 로드하여 재사용)
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
            self.clip_model.eval()  # 평가 모드로 설정
            
            self.is_initialized = True
            logger.info("FastSAM and CLIP models loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load models: {e}")
    
    def decode_image(self, data):
        """WebSocket으로 받은 이미지 데이터 디코딩"""
        try:
            # Color image (compressed, base64)
            color_info = data.get('color_image_compressed', {})
            color_b64 = color_info.get('data_base64', '')
            color_format = color_info.get('format', 'jpeg')
            
            if not color_b64:
                logger.error("No color image data in message")
                return None
            
            # Base64 디코딩
            color_bytes = base64.b64decode(color_b64)
            color_array = np.frombuffer(color_bytes, dtype=np.uint8)
            
            # 이미지 디코딩 (JPEG 또는 PNG)
            color_image = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
            
            if color_image is None:
                logger.error("Failed to decode color image")
                return None
            
            # BGR to RGB 변환 (PIL은 RGB 사용)
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"[DEBUG] Color image decoded: {color_image_rgb.shape}, dtype: {color_image_rgb.dtype}")
            
            return color_image_rgb
            
        except Exception as e:
            logger.error(f"[ERROR] 이미지 디코딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def process_frame(self, color_image, text_input):
        """단일 프레임 처리 - FastSAM 사용"""
        try:
            import time
            start_time = time.time()
            
            if not self.is_initialized:
                logger.error("FastSAM model not initialized")
                return None
            
            # numpy array를 PIL Image로 변환
            pil_image = Image.fromarray(color_image).convert("RGB")
            logger.info(f"Processing image with text prompt: '{text_input}'")
            
            # FastSAM 추론 실행
            everything_results = self.model(
                pil_image,
                device=self.device,
                retina_masks=self.config.get('retina', True),
                imgsz=self.config.get('imgsz', 1024),
                conf=self.config.get('conf', 0.4),
                iou=self.config.get('iou', 0.9)
            )
            
            # 최적화된 FastSAMPrompt 사용 (CLIP 모델 재사용)
            prompt_process = OptimizedFastSAMPrompt(
                pil_image, 
                everything_results, 
                device=self.device,
                clip_model=self.clip_model,
                clip_preprocess=self.clip_preprocess
            )
            
            # Text prompt로 segmentation 수행
            ann = prompt_process.text_prompt(text=text_input)
            
            # 결과 처리
            # ann은 numpy array 형태: [N, H, W]
            if ann is None or len(ann) == 0:
                logger.warning(f"No segmentation result for text: '{text_input}'")
                return {
                    'status': 'success',
                    'text_input': text_input,
                    'mask_image_base64': None,
                    'mask_path': None,
                    'normalized_obj_center': None,
                    'message': f"No object found for '{text_input}'",
                    'processing_time': time.time() - start_time
                }
            
            # 마스크가 모두 False인지 확인
            if isinstance(ann, np.ndarray) and not np.any(ann):
                logger.warning(f"Empty mask for text: '{text_input}'")
                return {
                    'status': 'success',
                    'text_input': text_input,
                    'mask_image_base64': None,
                    'mask_path': None,
                    'normalized_obj_center': None,
                    'message': f"Empty segmentation result for '{text_input}'",
                    'processing_time': time.time() - start_time
                }
            
            # 마스크 생성 및 Base64 인코딩
            mask_image_base64, mask_path, normalized_center = self._save_mask_results(
                ann, pil_image, text_input
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'text_input': text_input,
                'mask_image_base64': mask_image_base64,
                'mask_path': mask_path,
                'normalized_obj_center': normalized_center,
                'message': "Segmentation complete",
                'processing_time': processing_time
            }
            
            logger.info(f"Segmentation successful. Processing time: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] 프레임 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_mask_results(self, annotations, pil_image, text_input):
        """마스크 결과를 저장하고 Base64로 인코딩"""
        try:
            # annotations는 numpy array 형태: shape [N, H, W]
            # text_prompt에서 np.array([mask])로 반환됨
            if isinstance(annotations, np.ndarray):
                # 첫 번째 마스크 선택
                mask = annotations[0]
                logger.info(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
                
                # 마스크에 True 값이 있는지 확인
                if not np.any(mask):
                    logger.warning("Mask is empty (all False)")
                    return None, None, None
                    
            elif hasattr(annotations, 'masks') and annotations.masks is not None:
                # annotations.masks shape: [N, H, W]
                mask = annotations.masks[0].cpu().numpy()  # 첫 번째 마스크 선택
            else:
                logger.warning("No masks found in annotations")
                return None, None, None
            
            # 마스크를 0-255 범위로 변환
            # boolean 타입이면 255로, 아니면 기존 방식 사용
            if mask.dtype == bool:
                mask_uint8 = (mask.astype(np.uint8) * 255)
            else:
                mask_uint8 = (mask * 255).astype(np.uint8)
            
            # PIL Image로 변환
            mask_image = Image.fromarray(mask_uint8)
            
            # Base64 인코딩
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 파일로 저장
            safe_text_input = "".join(c for c in text_input if c.isalnum() or c in " _-").rstrip()
            mask_filename = f"mask_{safe_text_input}.png"
            mask_path = os.path.join(self.config['output_dir'], mask_filename)
            mask_image.save(mask_path)
            # logger.info(f"Saved mask to: {mask_path}")
            
            # Segment image 생성 (원본 RGB 이미지에 마스크 적용)
            # 원본 이미지를 numpy 배열로 변환
            rgb_array = np.array(pil_image)
            # 마스크를 0-1 범위로 정규화 (3채널로 확장)
            mask_normalized = (mask.astype(np.float32) if mask.dtype == bool else mask.astype(np.float32))
            if mask_normalized.max() > 1.0:
                mask_normalized = mask_normalized / 255.0
            mask_3d = np.stack([mask_normalized] * 3, axis=-1)
            # 마스크 영역만 추출 (마스크가 0인 부분은 검은색)
            segment_array = (rgb_array * mask_3d).astype(np.uint8)
            segment_image = Image.fromarray(segment_array)
            
            # Segment image 저장 (덮어쓰기)
            segment_filename = f"segment_{safe_text_input}.png"
            segment_path = os.path.join(self.config['output_dir'], segment_filename)
            segment_image.save(segment_path)
            # logger.info(f"Saved segment image to: {segment_path}")
            
            # 마스크의 중심점 계산
            normalized_center = self._calculate_mask_center(mask, pil_image.width, pil_image.height)
            
            return mask_image_base64, mask_path, normalized_center
            
        except Exception as e:
            logger.error(f"Error saving mask results: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _calculate_mask_center(self, mask, image_width, image_height):
        """마스크의 중심점을 계산 (distance transform 사용)"""
        try:
            # 마스크가 비어있는지 확인
            if np.count_nonzero(mask) == 0:
                logger.warning("Mask is empty, cannot find center")
                return None
            
            # uint8로 변환 (boolean이면 True=1, False=0으로 변환)
            if mask.dtype == bool:
                mask_uint8 = mask.astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)
            
            # 거리 변환을 사용하여 가장 안쪽 지점 찾기
            dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            center_x, center_y = max_loc
            
            # 정규화된 좌표 계산
            normalized_center = (float(center_x / image_width), float(center_y / image_height))
            
            logger.info(f"Calculated normalized center: {normalized_center}")
            
            return normalized_center
            
        except Exception as e:
            logger.error(f"Error calculating mask center: {e}")
            return None


class DepthClient:
    """외부 서버로 depth 정보를 전송하는 클라이언트"""
    
    def __init__(self, server_uri: str, reconnect_interval: float = 5.0):
        self.server_uri = server_uri
        self.reconnect_interval = reconnect_interval
        self.websocket = None
        self.is_connected = False
        self.send_queue = asyncio.Queue()
        self._task = None
    
    async def start(self):
        """클라이언트 시작 및 재연결 태스크 시작"""
        self._task = asyncio.create_task(self._run_with_reconnect())
        logger.info(f"[DEPTH] Client started, connecting to {self.server_uri}")
    
    async def stop(self):
        """클라이언트 중지"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("[DEPTH] Client stopped")
    
    async def _run_with_reconnect(self):
        """재연결 로직이 포함된 메인 루프"""
        while True:
            try:
                await self._connect_and_process()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DEPTH] Connection error: {e}")
            
            if not self.is_connected:
                logger.info(f"[DEPTH] Reconnecting in {self.reconnect_interval} seconds...")
                await asyncio.sleep(self.reconnect_interval)
    
    async def _connect_and_process(self):
        """서버에 연결하고 메시지 처리"""
        try:
            async with websockets.connect(
                self.server_uri,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                self.websocket = websocket
                self.is_connected = True
                logger.info(f"[DEPTH] Connected to {self.server_uri}")
                
                # 메시지 전송 태스크 시작
                send_task = asyncio.create_task(self._send_worker())
                
                try:
                    # 연결 유지 및 메시지 수신 대기
                    async for message in websocket:
                        # 서버로부터 메시지를 받을 경우 처리 (필요시)
                        logger.debug(f"[DEPTH] Received message: {message}")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("[DEPTH] Connection closed by server")
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass
        finally:
            self.is_connected = False
            self.websocket = None
    
    async def _send_worker(self):
        """큐에서 메시지를 가져와서 전송하는 워커"""
        while self.is_connected:
            try:
                message = await asyncio.wait_for(self.send_queue.get(), timeout=1.0)
                if self.websocket and self.is_connected:
                    try:
                        await self.websocket.send(message)
                        logger.debug("[DEPTH] Message sent successfully")
                    except Exception as e:
                        logger.error(f"[DEPTH] Error sending message: {e}")
                        # 전송 실패한 메시지를 다시 큐에 넣지 않음
                else:
                    logger.warning("[DEPTH] Not connected, message dropped")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    async def send(self, message: dict):
        """메시지를 큐에 추가하여 전송"""
        if not self.is_connected:
            logger.warning("[DEPTH] Not connected, message queued")
        
        message_str = json.dumps(message)
        await self.send_queue.put(message_str)


# 전역 Depth 클라이언트 인스턴스
depth_client = None


async def handler(websocket, evaluator):
    """WebSocket 클라이언트 처리"""
    client_addr = websocket.remote_address
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected from {client_addr}")
    
    # 기본 텍스트 입력 (클라이언트가 제공하지 않을 경우)
    current_text_input = "pepsi can"
    
    # 프레임 큐 (최신 프레임만 유지, maxsize=1)
    frame_queue = asyncio.Queue(maxsize=1)
    processing_task = None
    
    async def process_frame_worker():
        """프레임 처리 워커 - 큐에서 프레임을 가져와 처리"""
        while True:
            try:
                # 큐에서 프레임 데이터 가져오기
                frame_data = await frame_queue.get()
                if frame_data is None:  # 종료 신호
                    frame_queue.task_done()
                    break
                
                data, timestamp, frame_count, text_input = frame_data
                
                try:
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Processing frame {frame_count}...")
                    
                    # 이미지 디코딩
                    color_image = evaluator.decode_image(data)
                    
                    if color_image is not None:
                        # SAM 처리
                        logger.info(f"  - Image size: {color_image.shape}")
                        result = evaluator.process_frame(color_image, text_input)
                        
                        if result:
                            # 결과 전송
                            response = {
                                'status': 'success',
                                'timestamp': timestamp,
                                'frame_count': frame_count,
                                'result': result
                            }
                            await websocket.send(json.dumps(response))
                            
                            logger.info(f"  - Segmentation complete for '{text_input}'")
                            logger.info(f"  - Processing time: {result.get('processing_time', 0):.3f}s")
                            
                            if result.get('normalized_obj_center'):
                                center = result['normalized_obj_center']
                                logger.info(f"  - Object center (normalized): ({center[0]:.4f}, {center[1]:.4f})")
                            else:
                                logger.info(f"  - Message: {result.get('message', 'N/A')}")
                            
                            # mask_image_base64가 있고 depth 정보가 있으면 외부 서버로 전송
                            if result.get('mask_image_base64') and result.get('status') == 'success':
                                depth_info = data.get('depth_image_compressed')
                                depth_camera_info = data.get('depth_camera_info')
                                
                                if depth_info or depth_camera_info:
                                    broadcast_message = {
                                        'timestamp': timestamp,
                                        'frame_count': frame_count,
                                        'mask_image_base64': result.get('mask_image_base64'),
                                        'text_input': result.get('text_input'),
                                        'normalized_obj_center': result.get('normalized_obj_center'),
                                        'depth_image_compressed': depth_info,
                                        'depth_camera_info': depth_camera_info
                                    }
                                    
                                    if depth_client:
                                        await depth_client.send(broadcast_message)
                                        logger.info(f"  - Sent mask + depth data to external server")
                                    else:
                                        logger.warning("  - Depth client not initialized, message dropped")
                        else:
                            await websocket.send(json.dumps({
                                'status': 'error', 
                                'message': 'Processing failed'
                            }))
                    else:
                        await websocket.send(json.dumps({
                            'status': 'error', 
                            'message': 'Invalid image data'
                        }))
                    
                    logger.info("-" * 50)
                finally:
                    frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"[ERROR] Frame processing error: {e}")
                import traceback
                traceback.print_exc()
                if 'frame_data' in locals() and frame_data is not None:
                    frame_queue.task_done()
    
    try:
        # 프레임 처리 워커 시작
        processing_task = asyncio.create_task(process_frame_worker())
        
        async for message in websocket:
            try:
                data = json.loads(message)
                timestamp = data.get('timestamp', 'N/A')
                frame_count = data.get('frame_count', 'N/A')
                
                # 텍스트 입력 업데이트 (있는 경우)
                if 'text_input' in data:
                    current_text_input = data['text_input']
                    logger.info(f"  - Text input updated: '{current_text_input}'")
                
                # 프레임 큐에 추가 (큐가 가득 차면 오래된 프레임은 자동으로 버려짐)
                try:
                    frame_queue.put_nowait((data, timestamp, frame_count, current_text_input))
                    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Frame {frame_count} queued")
                except asyncio.QueueFull:
                    # 큐가 가득 찬 경우 (이전 프레임이 아직 처리 중), 오래된 프레임을 버리고 최신 프레임 추가
                    try:
                        old_frame = frame_queue.get_nowait()  # 오래된 프레임 제거
                        logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Skipped old frame {old_frame[2]}, queuing new frame {frame_count}")
                        frame_queue.put_nowait((data, timestamp, frame_count, current_text_input))
                    except asyncio.QueueEmpty:
                        # 큐가 비어있으면 그냥 추가
                        frame_queue.put_nowait((data, timestamp, frame_count, current_text_input))
                
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] JSON 디코딩 실패: {e}")
                await websocket.send(json.dumps({
                    'status': 'error', 
                    'message': 'Invalid JSON'
                }))
            except Exception as e:
                logger.error(f"[ERROR] 메시지 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({
                    'status': 'error', 
                    'message': str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Client {client_addr} disconnected")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error with client {client_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 워커 종료
        if processing_task:
            await frame_queue.put(None)  # 종료 신호
            await processing_task


async def main():
    """메인 서버 실행"""
    global depth_client
    
    # Configuration
    config = {
        'fastsam_model_path': "/app/weights/FastSAM.pt",
        'temp_dir': "/app/temp_data",
        # 'output_dir': "/app/outputs",
        'output_dir': "/app",
        'imgsz': 1024,
        'iou': 0.9,
        'conf': 0.4,
        'retina': True
    }
    
    # Depth 서버 URI (환경변수 또는 기본값)
    depth_server_uri = os.getenv('DEPTH_SERVER_URI', 'ws://localhost:8767')
    
    # Evaluator 초기화
    evaluator = FastSAMEvaluator(config)
    
    # Depth 클라이언트 초기화 및 시작
    depth_client = DepthClient(depth_server_uri, reconnect_interval=5.0)
    await depth_client.start()
    
    try:
        # WebSocket 서버 시작 (8766 포트: FastSAM 처리 서버)
        async with websockets.serve(
            lambda ws: handler(ws, evaluator),
            "0.0.0.0",
            8766,
            max_size=10_000_000,
            ping_interval=20,
            ping_timeout=10
        ):
            print("=" * 60)
            print("FastSAM Evaluation Server started at ws://0.0.0.0:8766")
            print(f"Depth Client connecting to {depth_server_uri}")
            print("=" * 60)
            print("Configuration:")
            print(f"  - FastSAM Model: {config['fastsam_model_path']}")
            print(f"  - Image Size: {config['imgsz']}")
            print(f"  - Confidence Threshold: {config['conf']}")
            print(f"  - IOU Threshold: {config['iou']}")
            print(f"  - Output Directory: {config['output_dir']}")
            print(f"  - Depth Server URI: {depth_server_uri}")
            print("=" * 60)
            print("Waiting for connections...")
            print("=" * 60)
            await asyncio.Future()  # run forever
    finally:
        # 클라이언트 정리
        if depth_client:
            await depth_client.stop()


if __name__ == "__main__":
    asyncio.run(main())

