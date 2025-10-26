# -*- coding: utf-8 -*-
"""
Heatmap Analysis Service
วิเคราะห์ Grad-CAM heatmap เพื่อระบุบริเวณที่โมเดลตรวจพบความผิดปกติ
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from PIL import Image


class HeatmapAnalyzer:
    """วิเคราะห์ Heatmap และระบุบริเวณที่มีความผิดปกติ"""

    # กำหนดบริเวณต่างๆ ของใบหน้าตามตำแหน่ง (ใช้กับภาพ 224x224)
    FACE_REGIONS = {
        'forehead': {
            'bbox': (50, 20, 174, 80),  # (x1, y1, x2, y2)
            'name_th': 'หน้าผาก',
            'name_en': 'Forehead'
        },
        'left_eye': {
            'bbox': (50, 70, 100, 110),
            'name_th': 'ตาซ้าย',
            'name_en': 'Left Eye'
        },
        'right_eye': {
            'bbox': (124, 70, 174, 110),
            'name_th': 'ตาขวา',
            'name_en': 'Right Eye'
        },
        'nose': {
            'bbox': (90, 100, 134, 150),
            'name_th': 'จมูก',
            'name_en': 'Nose'
        },
        'mouth': {
            'bbox': (80, 150, 144, 190),
            'name_th': 'ปาก',
            'name_en': 'Mouth'
        },
        'left_cheek': {
            'bbox': (30, 110, 80, 160),
            'name_th': 'แก้มซ้าย',
            'name_en': 'Left Cheek'
        },
        'right_cheek': {
            'bbox': (144, 110, 194, 160),
            'name_th': 'แก้มขวา',
            'name_en': 'Right Cheek'
        },
        'chin': {
            'bbox': (80, 180, 144, 220),
            'name_th': 'คาง',
            'name_en': 'Chin'
        },
        'face_boundary': {
            'bbox': (20, 60, 40, 180),  # ขอบใบหน้าซ้าย
            'bbox_right': (184, 60, 204, 180),  # ขอบใบหน้าขวา
            'name_th': 'ขอบใบหน้า',
            'name_en': 'Face Boundary'
        }
    }

    # เกณฑ์การตัดสิน
    HIGH_ATTENTION_THRESHOLD = 0.6  # ค่าสูงกว่านี้ = พบความผิดปกติ
    MODERATE_ATTENTION_THRESHOLD = 0.4  # ค่าปานกลาง

    def __init__(self):
        pass

    def analyze_heatmap(self, heatmap: np.ndarray, is_fake: bool) -> Dict:
        """
        วิเคราะห์ heatmap และระบุบริเวณที่มีความผิดปกติ

        Args:
            heatmap: Grad-CAM heatmap (224x224)
            is_fake: ผลการทำนายว่าเป็น FAKE หรือไม่

        Returns:
            ข้อมูลการวิเคราะห์พร้อมบริเวณที่ตรวจพบ
        """
        # Ensure heatmap is 2D numpy array
        if isinstance(heatmap, Image.Image):
            heatmap = np.array(heatmap.convert('L')) / 255.0
        elif len(heatmap.shape) == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY) / 255.0

        # Resize to 224x224 if needed
        if heatmap.shape != (224, 224):
            heatmap = cv2.resize(heatmap, (224, 224))

        # วิเคราะห์แต่ละบริเวณ
        region_analysis = []

        for region_id, region_info in self.FACE_REGIONS.items():
            # คำนวณค่าความสนใจเฉลี่ยในบริเวณนี้
            if region_id == 'face_boundary':
                # กรณีขอบใบหน้า ตรวจทั้ง 2 ด้าน
                bbox_left = region_info['bbox']
                bbox_right = region_info['bbox_right']

                attention_left = self._get_region_attention(heatmap, bbox_left)
                attention_right = self._get_region_attention(heatmap, bbox_right)
                avg_attention = (attention_left + attention_right) / 2
                max_attention = max(attention_left, attention_right)
            else:
                bbox = region_info['bbox']
                avg_attention = self._get_region_attention(heatmap, bbox)
                max_attention = self._get_max_attention(heatmap, bbox)

            # จัดระดับความสนใจ
            attention_level = self._classify_attention(avg_attention)

            region_analysis.append({
                'region_id': region_id,
                'region_name_th': region_info['name_th'],
                'region_name_en': region_info['name_en'],
                'avg_attention': float(avg_attention),
                'max_attention': float(max_attention),
                'attention_level': attention_level,
                'bbox': region_info.get('bbox', None)
            })

        # เรียงลำดับตามความสนใจ (สูง -> ต่ำ)
        region_analysis.sort(key=lambda x: x['avg_attention'], reverse=True)

        # หาบริเวณที่น่าสงสัยมากที่สุด (สำหรับ FAKE)
        suspicious_regions = []
        if is_fake:
            suspicious_regions = [
                r for r in region_analysis
                if r['avg_attention'] >= self.HIGH_ATTENTION_THRESHOLD
            ]

        # สร้างคำอธิบาย
        explanation = self._generate_explanation(
            region_analysis,
            suspicious_regions,
            is_fake
        )

        # หาจุดที่มีความสนใจสูงสุด
        hotspot = self._find_hotspot(heatmap)

        return {
            'is_fake': is_fake,
            'regions': region_analysis,
            'suspicious_regions': suspicious_regions,
            'top_3_regions': region_analysis[:3],
            'explanation': explanation,
            'hotspot': hotspot,
            'overall_attention': float(np.mean(heatmap)),
            'max_attention_value': float(np.max(heatmap))
        }

    def _get_region_attention(self, heatmap: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """คำนวณค่าความสนใจเฉลี่ยในบริเวณที่กำหนด"""
        x1, y1, x2, y2 = bbox
        region = heatmap[y1:y2, x1:x2]
        return np.mean(region)

    def _get_max_attention(self, heatmap: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """คำนวณค่าความสนใจสูงสุดในบริเวณที่กำหนด"""
        x1, y1, x2, y2 = bbox
        region = heatmap[y1:y2, x1:x2]
        return np.max(region)

    def _classify_attention(self, attention: float) -> str:
        """จัดระดับความสนใจ"""
        if attention >= self.HIGH_ATTENTION_THRESHOLD:
            return 'high'
        elif attention >= self.MODERATE_ATTENTION_THRESHOLD:
            return 'moderate'
        else:
            return 'low'

    def _find_hotspot(self, heatmap: np.ndarray) -> Dict:
        """หาจุดที่มีความสนใจสูงสุด"""
        max_idx = np.argmax(heatmap)
        y, x = np.unravel_index(max_idx, heatmap.shape)

        return {
            'x': int(x),
            'y': int(y),
            'value': float(heatmap[y, x])
        }

    def _generate_explanation(
        self,
        region_analysis: List[Dict],
        suspicious_regions: List[Dict],
        is_fake: bool
    ) -> Dict:
        """สร้างคำอธิบายผลการวิเคราะห์"""

        if is_fake and len(suspicious_regions) > 0:
            # กรณี FAKE - อธิบายบริเวณที่น่าสงสัย
            top_region = suspicious_regions[0]

            # คำอธิบายเฉพาะแต่ละบริเวณ
            region_explanations = {
                'mouth': 'บริเวณปากมักพบความผิดปกติใน Deepfake เนื่องจากการสร้างภาพปากขณะพูดที่ไม่เป็นธรรมชาติ หรือฟันที่ผิดปกติ',
                'left_eye': 'บริเวณตามักมีปัญหาเรื่องแสงสะท้อน, ขนตา, หรือการกระพริบตาที่ไม่เป็นธรรมชาติ',
                'right_eye': 'บริเวณตามักมีปัญหาเรื่องแสงสะท้อน, ขนตา, หรือการกระพริบตาที่ไม่เป็นธรรมชาติ',
                'nose': 'บริเวณจมูกอาจมีการเชื่อมต่อกับใบหน้าที่ไม่ราบรื่น หรือเงาที่ผิดปกติ',
                'face_boundary': 'ขอบใบหน้ามักเป็นจุดที่พบ artifact จากการผสมภาพ (blending) ที่ไม่สมบูรณ์',
                'left_cheek': 'บริเวณแก้มอาจมีความไม่สม่ำเสมอของผิวหนัง หรือการเชื่อมต่อที่ไม่เป็นธรรมชาติ',
                'right_cheek': 'บริเวณแก้มอาจมีความไม่สม่ำเสมอของผิวหนัง หรือการเชื่อมต่อที่ไม่เป็นธรรมชาติ',
                'chin': 'บริเวณคางมักมีปัญหาเรื่องการเชื่อมต่อกับคอ หรือเงาที่ผิดปกติ',
                'forehead': 'บริเวณหน้าผากอาจมีการเปลี่ยนแปลงของ texture หรือแสงที่ไม่สม่ำเสมอ'
            }

            specific_explanation = region_explanations.get(
                top_region['region_id'],
                'พบความผิดปกติในบริเวณนี้'
            )

            summary_th = (
                f"โมเดลตรวจพบความผิดปกติสูงสุดที่ **{top_region['region_name_th']}** "
                f"(ระดับความสนใจ: {top_region['avg_attention']:.1%})"
            )

            details_th = [
                f"🔴 **{r['region_name_th']}**: ระดับความสนใจ {r['avg_attention']:.1%} - {region_explanations.get(r['region_id'], '')}"
                for r in suspicious_regions[:3]
            ]

            summary_en = (
                f"Model detected highest anomaly at **{top_region['region_name_en']}** "
                f"(attention level: {top_region['avg_attention']:.1%})"
            )

        elif is_fake:
            # กรณี FAKE แต่ไม่มีบริเวณที่เด่นชัด
            top_3 = region_analysis[:3]
            summary_th = (
                f"โมเดลตรวจพบความผิดปกติกระจายทั่วใบหน้า "
                f"โดยเฉพาะที่ {', '.join([r['region_name_th'] for r in top_3])}"
            )
            details_th = [
                f"⚠️ **{r['region_name_th']}**: ระดับความสนใจ {r['avg_attention']:.1%}"
                for r in top_3
            ]
            summary_en = (
                f"Model detected distributed anomalies across face, "
                f"especially at {', '.join([r['region_name_en'] for r in top_3])}"
            )
            specific_explanation = "ความผิดปกติกระจายทั่วใบหน้า ไม่เฉพาะเจาะจงบริเวณใดบริเวณหนึ่ง"

        else:
            # กรณี REAL
            top_3 = region_analysis[:3]
            summary_th = (
                f"โมเดลพบว่าภาพนี้เป็นธรรมชาติ "
                f"การกระจายความสนใจปกติที่ {', '.join([r['region_name_th'] for r in top_3])}"
            )
            details_th = [
                f"✅ **{r['region_name_th']}**: ระดับความสนใจ {r['avg_attention']:.1%} (ปกติ)"
                for r in top_3
            ]
            summary_en = (
                f"Model confirms natural image. "
                f"Normal attention distribution at {', '.join([r['region_name_en'] for r in top_3])}"
            )
            specific_explanation = "ภาพแสดงลักษณะที่เป็นธรรมชาติ ไม่พบสัญญาณของการปลอมแปลง"

        return {
            'summary_th': summary_th,
            'summary_en': summary_en,
            'details_th': details_th,
            'specific_explanation': specific_explanation
        }

    def create_annotated_regions(self, image_size: Tuple[int, int] = (224, 224)) -> List[Dict]:
        """
        สร้างข้อมูล bounding boxes สำหรับแสดงบริเวณต่างๆ บน UI

        Returns:
            รายการ regions พร้อม coordinates สำหรับวาดบน canvas
        """
        regions = []

        for region_id, region_info in self.FACE_REGIONS.items():
            if region_id == 'face_boundary':
                # แยกเป็น 2 boxes
                regions.append({
                    'id': 'face_boundary_left',
                    'name_th': 'ขอบใบหน้าซ้าย',
                    'name_en': 'Left Boundary',
                    'bbox': region_info['bbox']
                })
                regions.append({
                    'id': 'face_boundary_right',
                    'name_th': 'ขอบใบหน้าขวา',
                    'name_en': 'Right Boundary',
                    'bbox': region_info['bbox_right']
                })
            else:
                regions.append({
                    'id': region_id,
                    'name_th': region_info['name_th'],
                    'name_en': region_info['name_en'],
                    'bbox': region_info['bbox']
                })

        return regions
