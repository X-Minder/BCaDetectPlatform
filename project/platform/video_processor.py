"""
video_processor.py
膀胱镜视频处理模块
解决Web浏览器播放兼容性问题
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import os
import subprocess
import tempfile
import shutil


class TemporalSmoother:
    """时序平滑器 - 处理帧间抖动问题"""

    def __init__(self, window_size=5, method='exponential'):
        self.window_size = window_size
        self.method = method
        self.prob_history = deque(maxlen=window_size)
        self.mask_history = deque(maxlen=window_size)
        self.grade_history = deque(maxlen=window_size)
        self.alpha = 2 / (window_size + 1)

    def smooth_probability(self, prob: float) -> float:
        self.prob_history.append(prob)
        if len(self.prob_history) < 2:
            return prob

        if self.method == 'moving_average':
            return np.mean(self.prob_history)
        elif self.method == 'exponential':
            smoothed = self.prob_history[0]
            for p in list(self.prob_history)[1:]:
                smoothed = self.alpha * p + (1 - self.alpha) * smoothed
            return smoothed
        elif self.method == 'median':
            return np.median(self.prob_history)
        return prob

    def smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        self.mask_history.append(mask_cleaned)
        if len(self.mask_history) < 2:
            return mask_cleaned

        if self.method == 'moving_average':
            masks_array = np.array(list(self.mask_history), dtype=np.float32)
            smoothed_mask = np.mean(masks_array, axis=0)
            smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255
        elif self.method == 'exponential':
            smoothed_mask = self.mask_history[0].astype(np.float32)
            for m in list(self.mask_history)[1:]:
                smoothed_mask = self.alpha * m + (1 - self.alpha) * smoothed_mask
            smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255
        elif self.method == 'median':
            masks_array = np.array(list(self.mask_history), dtype=np.uint8)
            smoothed_mask = np.median(masks_array, axis=0).astype(np.uint8)
        else:
            smoothed_mask = mask_cleaned

        smoothed_mask = cv2.GaussianBlur(smoothed_mask, (5, 5), 1.0)
        smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255
        return smoothed_mask

    def smooth_grade(self, grade: int) -> int:
        if grade is None:
            return None
        self.grade_history.append(grade)
        if len(self.grade_history) < 3:
            return grade
        grades = list(self.grade_history)
        return max(set(grades), key=grades.count)

    def reset(self):
        self.prob_history.clear()
        self.mask_history.clear()
        self.grade_history.clear()


class CystoscopyVideoProcessor:
    """膀胱镜视频处理器"""

    def __init__(self,
                 classification_model,
                 segmentation_model,
                 grading_model,
                 preprocess_transform,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 enable_temporal_smoothing=True,
                 smoothing_window=5,
                 smoothing_method='exponential'):
        self.device = device
        self.cls_model = classification_model
        self.seg_model = segmentation_model
        self.grade_model = grading_model
        self.preprocess = preprocess_transform

        self.enable_smoothing = enable_temporal_smoothing
        if self.enable_smoothing:
            self.smoother = TemporalSmoother(
                window_size=smoothing_window,
                method=smoothing_method
            )
            print(f"✓ 启用时序平滑: 窗口={smoothing_window}, 方法={smoothing_method}")

        self.font_cn = self._load_chinese_font()

    def _load_chinese_font(self, font_size=32):
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]

        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"✓ 加载中文字体: {font_path}")
                    return font
            except Exception:
                continue

        print("⚠ 未找到中文字体,将使用英文显示")
        return None

    def _put_chinese_text(self, img, text, position, font_size=32, color=(255, 255, 255)):
        if self.font_cn is None:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, color, 2, cv2.LINE_AA)
            return img

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font_cn, fill=color)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = self.preprocess(pil_img).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray,
                      cls_threshold: float = 0.5,
                      enable_classification: bool = True,
                      enable_segmentation: bool = True,
                      enable_grading: bool = True) -> Dict:
        tensor = self.preprocess_frame(frame)
        h, w = frame.shape[:2]

        result = {
            'is_tumor': False,
            'tumor_prob': 0.0,
            'tumor_prob_raw': 0.0,
            'mask': None,
            'mask_raw': None,
            'grade': None,
            'grade_raw': None,
            'grade_label': '-'
        }

        if enable_classification and self.cls_model is not None:
            cls_output = self.cls_model(tensor)
            tumor_prob_raw = torch.sigmoid(cls_output).item()
            result['tumor_prob_raw'] = tumor_prob_raw

            if self.enable_smoothing:
                tumor_prob = self.smoother.smooth_probability(tumor_prob_raw)
            else:
                tumor_prob = tumor_prob_raw

            result['tumor_prob'] = tumor_prob
            result['is_tumor'] = tumor_prob > cls_threshold

        should_segment = enable_segmentation and (
                not enable_classification or result['is_tumor']
        )

        if should_segment and self.seg_model is not None:
            seg_outputs = self.seg_model(tensor)

            if isinstance(seg_outputs, list):
                seg_out = seg_outputs[0]
            else:
                seg_out = seg_outputs

            mask_values = torch.sigmoid(seg_out).squeeze().cpu().numpy()

            if np.any(mask_values > 0.5):
                mask_binary = (mask_values > 0.5).astype(np.uint8) * 255
                mask_resized = cv2.resize(mask_binary, (w, h),
                                          interpolation=cv2.INTER_NEAREST)

                if self.enable_smoothing:
                    mask_smoothed = self.smoother.smooth_mask(mask_resized)
                else:
                    mask_smoothed = mask_resized

                result['mask'] = mask_smoothed

        should_grade = enable_grading and (
                not enable_classification or result['is_tumor']
        )

        if should_grade and self.grade_model is not None:
            grade_output = self.grade_model(tensor)
            grade_prob_tensor = torch.softmax(grade_output, dim=1)
            grade_idx = torch.argmax(grade_prob_tensor).item()

            if self.enable_smoothing:
                grade_smoothed = self.smoother.smooth_grade(grade_idx)
            else:
                grade_smoothed = grade_idx

            result['grade'] = grade_smoothed
            result['grade_label'] = 'High Grade' if grade_idx == 1 else 'Low Grade'

        return result

    def annotate_frame(self, frame: np.ndarray,
                       prediction: Dict,
                       show_mask: bool = True,
                       mask_alpha: float = 0.4,
                       lang: str = 'English',
                       show_smoothing_indicator: bool = True) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        tumor_prob = prediction['tumor_prob']
        is_tumor = prediction['is_tumor']

        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (420, 130), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        if lang == '中文':
            status_text = "肿瘤" if is_tumor else "正常"
            prob_text = f"概率: {tumor_prob * 100:.1f}%"
            status_color = (0, 0, 255) if is_tumor else (0, 255, 0)
            use_chinese = True
        else:
            status_text = "TUMOR" if is_tumor else "NORMAL"
            prob_text = f"Prob: {tumor_prob * 100:.1f}%"
            status_color = (0, 0, 255) if is_tumor else (0, 255, 0)
            use_chinese = False

        if use_chinese and self.font_cn:
            annotated = self._put_chinese_text(
                annotated, status_text, (10, 15),
                font_size=40, color=status_color
            )
            annotated = self._put_chinese_text(
                annotated, prob_text, (10, 60),
                font_size=32, color=(255, 255, 255)
            )
        else:
            cv2.putText(annotated, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
            cv2.putText(annotated, prob_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if is_tumor and prediction['mask'] is not None and show_mask:
            mask = prediction['mask']

            mask_colored = np.zeros_like(frame)
            mask_colored[:, :, 2] = mask

            mask_bool = mask > 0
            annotated[mask_bool] = cv2.addWeighted(
                annotated[mask_bool], 1 - mask_alpha,
                mask_colored[mask_bool], mask_alpha, 0
            )

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 255, 255), 3)

            if prediction['grade_label'] != '-':
                if lang == '中文':
                    grade_text = f"分级: {prediction['grade_label']}"
                    if self.font_cn:
                        annotated = self._put_chinese_text(
                            annotated, grade_text, (10, 105),
                            font_size=32, color=(255, 165, 0)
                        )
                    else:
                        grade_text_en = f"Grade: {prediction['grade_label']}"
                        cv2.putText(annotated, grade_text_en, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                   (255, 165, 0), 2, cv2.LINE_AA)
                else:
                    grade_text = f"Grade: {prediction['grade_label']}"
                    cv2.putText(annotated, grade_text, (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                               (255, 165, 0), 2, cv2.LINE_AA)

        return annotated

    def _check_ffmpeg_available(self) -> bool:
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode == 0:
                version_info = result.stdout.decode('utf-8', errors='ignore').split('\n')[0]
                print(f"✓ FFmpeg可用: {version_info}")
                return True
            return False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"⚠ FFmpeg不可用: {e}")
            return False

    def _convert_with_ffmpeg(self, input_path: str, output_path: str) -> bool:
        try:
            if not os.path.exists(input_path):
                print(f"⚠ 输入文件不存在: {input_path}")
                return False

            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            print(f"执行FFmpeg转换...")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )

            if result.returncode != 0:
                stderr_output = result.stderr.decode('utf-8', errors='ignore')
                print(f"⚠ FFmpeg错误输出:\n{stderr_output[-500:]}")
                return False

            if not os.path.exists(output_path):
                print(f"⚠ FFmpeg转换后文件不存在: {output_path}")
                return False

            print(f"✓ FFmpeg转换成功: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return True

        except subprocess.TimeoutExpired:
            print(f"⚠ FFmpeg转换超时")
            return False
        except Exception as e:
            print(f"⚠ FFmpeg转换失败: {e}")
            return False

    def process_video(self,
                      input_path: str,
                      output_path: str,
                      cls_threshold: float = 0.5,
                      show_mask: bool = True,
                      mask_alpha: float = 0.4,
                      skip_frames: int = 0,
                      max_frames: Optional[int] = None,
                      enable_classification: bool = True,
                      enable_segmentation: bool = True,
                      enable_grading: bool = True,
                      lang: str = 'English',
                      progress_callback=None) -> Dict:
        """处理完整视频 - 确保Web浏览器兼容性"""

        if self.enable_smoothing:
            self.smoother.reset()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"视频信息: {width}x{height} @ {fps}fps, 共{total_frames}帧")

        use_ffmpeg = self._check_ffmpeg_available()

        # 创建独立的临时目录
        temp_dir = tempfile.mkdtemp(prefix='bladder_video_')
        print(f"✓ 创建临时目录: {temp_dir}")

        try:
            if use_ffmpeg:
                print("✓ 将使用FFmpeg进行最终编码")
                temp_output = os.path.join(temp_dir, 'temp_video.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                print("⚠ 未检测到FFmpeg,使用OpenCV直接编码")
                temp_output = output_path
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError("无法创建视频写入器")

            print(f"✓ OpenCV写入到: {temp_output}")

            stats = {
                'total_frames': 0,
                'processed_frames': 0,
                'tumor_frames': 0,
                'high_grade_frames': 0,
                'low_grade_frames': 0
            }

            frame_idx = 0
            pbar = tqdm(total=total_frames, desc="处理视频")

            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                stats['total_frames'] += 1

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    out.write(frame)
                    frame_idx += 1
                    pbar.update(1)
                    continue

                prediction = self.predict_frame(
                    frame, cls_threshold,
                    enable_classification,
                    enable_segmentation,
                    enable_grading
                )

                annotated_frame = self.annotate_frame(
                    frame, prediction, show_mask, mask_alpha, lang
                )

                out.write(annotated_frame)

                stats['processed_frames'] += 1
                if prediction['is_tumor']:
                    stats['tumor_frames'] += 1
                    if prediction['grade'] == 1:
                        stats['high_grade_frames'] += 1
                    elif prediction['grade'] == 0:
                        stats['low_grade_frames'] += 1

                if progress_callback:
                    progress = (frame_idx + 1) / total_frames
                    progress_callback(progress)

                frame_idx += 1
                pbar.update(1)

            cap.release()
            out.release()
            pbar.close()

            # 验证临时文件
            if not os.path.exists(temp_output):
                raise RuntimeError(f"临时视频文件创建失败: {temp_output}")

            temp_size = os.path.getsize(temp_output)
            if temp_size < 1000:
                raise RuntimeError(f"临时文件异常小 ({temp_size} bytes)")

            print(f"✓ OpenCV编码完成: {temp_size / 1024 / 1024:.2f} MB")

            # FFmpeg转码
            if use_ffmpeg:
                print("\n开始FFmpeg转码为Web兼容格式...")
                if self._convert_with_ffmpeg(temp_output, output_path):
                    print("✓ FFmpeg转码成功")
                else:
                    print("⚠ FFmpeg转码失败,复制原始文件")
                    shutil.copy2(temp_output, output_path)

            # 验证最终输出
            if not os.path.exists(output_path):
                raise RuntimeError(f"最终视频文件不存在: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                raise RuntimeError(f"输出视频文件异常小 ({file_size} bytes)")

            print(f"\n✓ 处理完成!")
            print(f"✓ 输出文件: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
            print(f"✓ 总帧数: {stats['total_frames']}")
            print(f"✓ 处理帧数: {stats['processed_frames']}")

            if stats['processed_frames'] > 0:
                tumor_rate = stats['tumor_frames'] / stats['processed_frames'] * 100
                print(f"✓ 肿瘤帧数: {stats['tumor_frames']} ({tumor_rate:.1f}%)")
                if stats['high_grade_frames'] + stats['low_grade_frames'] > 0:
                    print(f"✓ 高级别: {stats['high_grade_frames']}, 低级别: {stats['low_grade_frames']}")

            return stats

        finally:
            # 清理临时目录
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print("✓ 临时文件已清理")
            except Exception as e:
                print(f"⚠ 清理临时文件失败: {e}")