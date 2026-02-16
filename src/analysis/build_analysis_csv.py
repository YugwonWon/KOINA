#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_analysis_csv.py

원본 JSON 파일에서 메타데이터를 추출하고, 
TextGrid 파일에서 Points(pct) 정보를 파싱하여 
연구 분석을 위한 통합 CSV를 생성합니다.

확장 가능성을 위해 모듈화 설계되어 있습니다.
"""

import os
import sys
import json
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PitchPoint:
    """정규화된 피치 포인트 데이터"""
    time_pct: float  # 0-100% 정규화된 시간
    pitch_hz: float  # 피치 값 (Hz)


@dataclass
class UtteranceData:
    """발화 데이터"""
    # 식별자
    utterance_id: str
    file_id: str
    
    # 메타데이터
    text: str = ""
    style: str = ""
    sub_style: str = ""
    emotion: str = ""
    intensity: int = 0
    
    # 화자 정보
    speaker_gender: str = ""
    speaker_age: int = 0
    speaker_id: int = 0
    
    # 음성 정보
    duration: float = 0.0
    
    # 피치 데이터 (정규화된 퍼센트 기반)
    pitch_points_pct: List[Tuple[float, float]] = field(default_factory=list)
    
    # 피치 구간별 통계 (10% 구간)
    pitch_bin_0_10: float = 0.0
    pitch_bin_10_20: float = 0.0
    pitch_bin_20_30: float = 0.0
    pitch_bin_30_40: float = 0.0
    pitch_bin_40_50: float = 0.0
    pitch_bin_50_60: float = 0.0
    pitch_bin_60_70: float = 0.0
    pitch_bin_70_80: float = 0.0
    pitch_bin_80_90: float = 0.0
    pitch_bin_90_100: float = 0.0
    
    # 피치 포인트 수
    pitch_point_count: int = 0
    
    # 전체 피치 통계
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    pitch_range: float = 0.0


class TextGridParser:
    """TextGrid 파일 파서"""
    
    @staticmethod
    def parse_textgrid(filepath: str) -> Dict[str, Any]:
        """TextGrid 파일을 파싱하여 tier 정보를 반환"""
        result = {
            'points_pct': [],
            'points': [],
            'utterance': '',
            'duration': 0.0
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"TextGrid 파일 읽기 실패: {filepath}, {e}")
            return result
        
        # Points(pct) 티어 파싱
        points_pct_pattern = r'name = "Points\(pct\)".*?points: size = (\d+)(.*?)(?=item \[|$)'
        match = re.search(points_pct_pattern, content, re.DOTALL)
        
        if match:
            points_section = match.group(2)
            # 각 포인트 추출
            point_pattern = r'time = ([\d.]+)\s+mark = "([^"]*)"'
            for point_match in re.finditer(point_pattern, points_section):
                time_pct = float(point_match.group(1))
                try:
                    pitch_hz = float(point_match.group(2))
                    result['points_pct'].append((time_pct, pitch_hz))
                except ValueError:
                    continue
        
        # utterance 티어에서 발화 텍스트 추출
        utterance_pattern = r'name = "utterance".*?text = "([^"]*)"'
        match = re.search(utterance_pattern, content, re.DOTALL)
        if match:
            result['utterance'] = match.group(1).strip()
        
        # 발화 길이 추출 (utterance tier의 xmax)
        duration_pattern = r'item \[1\]:.*?xmax = ([\d.]+)'
        match = re.search(duration_pattern, content, re.DOTALL)
        if match:
            try:
                result['duration'] = float(match.group(1))
            except ValueError:
                pass
        
        return result


class MetadataExtractor:
    """JSON 메타데이터 추출기"""
    
    def __init__(self, json_dir: str):
        self.json_dir = Path(json_dir)
        self.metadata_cache: Dict[str, Dict] = {}
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """모든 JSON 파일에서 메타데이터 로드"""
        logger.info(f"메타데이터 로딩 중: {self.json_dir}")
        
        json_files = list(self.json_dir.rglob("*.json"))
        
        for json_file in tqdm(json_files, desc="JSON 메타데이터 로딩"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    file_id = item.get('id', '')
                    reciter = item.get('reciter', {})
                    
                    for sentence in item.get('sentences', []):
                        sent_id = sentence.get('id', '')
                        style_info = sentence.get('style', {})
                        voice_piece = sentence.get('voice_piece', {})
                        
                        self.metadata_cache[sent_id] = {
                            'file_id': file_id,
                            'text': sentence.get('origin_text', ''),
                            'style': style_info.get('style', ''),
                            'sub_style': style_info.get('sub_style', ''),
                            'emotion': style_info.get('emotion', ''),
                            'intensity': style_info.get('intensity', 0),
                            'speaker_gender': reciter.get('gender', ''),
                            'speaker_age': reciter.get('age', 0),
                            'speaker_id': reciter.get('id', 0),
                            'duration': voice_piece.get('duration', 0.0)
                        }
            except Exception as e:
                logger.warning(f"JSON 파일 처리 실패: {json_file}, {e}")
    
    def get_metadata(self, utterance_id: str) -> Optional[Dict]:
        """발화 ID로 메타데이터 조회"""
        return self.metadata_cache.get(utterance_id)


def calculate_bin_statistics(points_pct: List[Tuple[float, float]]) -> Dict[str, float]:
    """10% 구간별 평균 피치 계산"""
    bins = {f'pitch_bin_{i*10}_{(i+1)*10}': [] for i in range(10)}
    
    for time_pct, pitch_hz in points_pct:
        bin_idx = min(int(time_pct // 10), 9)
        bin_key = f'pitch_bin_{bin_idx*10}_{(bin_idx+1)*10}'
        bins[bin_key].append(pitch_hz)
    
    # 평균 계산 (데이터 없으면 0)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in bins.items()}


def linear_regression_slope(x: List[float], y: List[float]) -> Tuple[float, float]:
    """단순 선형 회귀로 기울기와 절편 계산
    
    Returns:
        (slope, intercept)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # 분산 및 공분산
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    if ss_xx == 0:
        return 0.0, mean_y
    
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    
    return slope, intercept


def calculate_slope_statistics(points_pct: List[Tuple[float, float]]) -> Dict[str, float]:
    """피치 기울기 관련 통계 계산
    
    - 전체 기울기 (선형 회귀)
    - 전반부/후반부 기울기
    - 기울기 변화량
    - 피치 속도 (연속 포인트 간 변화)
    """
    result = {
        'pitch_slope': 0.0,
        'pitch_slope_abs': 0.0,
        'pitch_slope_first_half': 0.0,
        'pitch_slope_second_half': 0.0,
        'pitch_slope_change': 0.0,
        'pitch_velocity_mean': 0.0,
        'pitch_velocity_std': 0.0,
        'pitch_velocity_abs_mean': 0.0,
        'pitch_acceleration_mean': 0.0,
        'pitch_inflection_count': 0,
        'pitch_peak_position': 0.0,
        'pitch_valley_position': 0.0,
    }
    
    if len(points_pct) < 2:
        return result
    
    # 시간순 정렬
    sorted_points = sorted(points_pct, key=lambda p: p[0])
    times = [p[0] for p in sorted_points]
    pitches = [p[1] for p in sorted_points]
    
    # 1. 전체 기울기 (선형 회귀)
    slope, _ = linear_regression_slope(times, pitches)
    result['pitch_slope'] = slope
    result['pitch_slope_abs'] = abs(slope)
    
    # 2. 전반부/후반부 기울기
    mid_idx = len(sorted_points) // 2
    if mid_idx >= 2:
        first_half = sorted_points[:mid_idx]
        second_half = sorted_points[mid_idx:]
        
        slope_first, _ = linear_regression_slope(
            [p[0] for p in first_half], 
            [p[1] for p in first_half]
        )
        slope_second, _ = linear_regression_slope(
            [p[0] for p in second_half], 
            [p[1] for p in second_half]
        )
        
        result['pitch_slope_first_half'] = slope_first
        result['pitch_slope_second_half'] = slope_second
        result['pitch_slope_change'] = slope_second - slope_first
    
    # 3. 피치 속도 (연속 포인트 간 기울기)
    velocities = []
    for i in range(1, len(sorted_points)):
        dt = times[i] - times[i-1]
        if dt > 0:
            velocity = (pitches[i] - pitches[i-1]) / dt
            velocities.append(velocity)
    
    if velocities:
        result['pitch_velocity_mean'] = sum(velocities) / len(velocities)
        result['pitch_velocity_abs_mean'] = sum(abs(v) for v in velocities) / len(velocities)
        
        vel_mean = result['pitch_velocity_mean']
        variance = sum((v - vel_mean) ** 2 for v in velocities) / len(velocities)
        result['pitch_velocity_std'] = variance ** 0.5
        
        # 4. 피치 가속도 (속도의 변화)
        if len(velocities) >= 2:
            accelerations = []
            for i in range(1, len(velocities)):
                # 시간 간격 근사
                dt = (times[i+1] - times[i-1]) / 2 if i+1 < len(times) else times[i] - times[i-1]
                if dt > 0:
                    acc = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acc)
            
            if accelerations:
                result['pitch_acceleration_mean'] = sum(accelerations) / len(accelerations)
        
        # 5. 굴곡점 수 (기울기 부호 변화 횟수)
        inflection_count = 0
        for i in range(1, len(velocities)):
            if velocities[i-1] * velocities[i] < 0:  # 부호 변화
                inflection_count += 1
        result['pitch_inflection_count'] = inflection_count
    
    # 6. 최고점/최저점 위치
    max_idx = pitches.index(max(pitches))
    min_idx = pitches.index(min(pitches))
    result['pitch_peak_position'] = times[max_idx]
    result['pitch_valley_position'] = times[min_idx]
    
    return result


def calculate_overall_statistics(points_pct: List[Tuple[float, float]]) -> Dict[str, float]:
    """전체 피치 통계 계산"""
    if not points_pct:
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_min': 0.0,
            'pitch_max': 0.0,
            'pitch_range': 0.0,
            'pitch_point_count': 0
        }
    
    pitches = [p[1] for p in points_pct]
    mean_val = sum(pitches) / len(pitches)
    variance = sum((p - mean_val) ** 2 for p in pitches) / len(pitches)
    std_val = variance ** 0.5
    min_val = min(pitches)
    max_val = max(pitches)
    
    return {
        'pitch_mean': mean_val,
        'pitch_std': std_val,
        'pitch_min': min_val,
        'pitch_max': max_val,
        'pitch_range': max_val - min_val,
        'pitch_point_count': len(pitches)
    }


def process_textgrid_file(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    """개별 TextGrid 파일 처리"""
    textgrid_path, utterance_id, metadata = args
    
    try:
        # TextGrid 파싱
        tg_data = TextGridParser.parse_textgrid(textgrid_path)
        points_pct = tg_data['points_pct']
        
        if not points_pct:
            return None
        
        # 통계 계산
        bin_stats = calculate_bin_statistics(points_pct)
        overall_stats = calculate_overall_statistics(points_pct)
        slope_stats = calculate_slope_statistics(points_pct)
        
        # 결과 생성
        result = {
            'utterance_id': utterance_id,
            'file_id': metadata.get('file_id', ''),
            'text': metadata.get('text', ''),
            'style': metadata.get('style', ''),
            'sub_style': metadata.get('sub_style', ''),
            'emotion': metadata.get('emotion', ''),
            'intensity': metadata.get('intensity', 0),
            'speaker_gender': metadata.get('speaker_gender', ''),
            'speaker_age': metadata.get('speaker_age', 0),
            'speaker_id': metadata.get('speaker_id', 0),
            'duration': metadata.get('duration', 0.0),
            'pitch_points_raw': json.dumps(points_pct),  # JSON 문자열로 저장
        }
        result.update(bin_stats)
        result.update(overall_stats)
        result.update(slope_stats)
        
        return result
        
    except Exception as e:
        logger.warning(f"TextGrid 처리 실패: {textgrid_path}, {e}")
        return None


def build_analysis_csv(
    json_dir: str,
    textgrid_dir: str,
    output_csv: str,
    num_workers: int = 8
):
    """분석용 CSV 생성 메인 함수"""
    
    # 메타데이터 로드
    metadata_extractor = MetadataExtractor(json_dir)
    logger.info(f"총 {len(metadata_extractor.metadata_cache)}개 메타데이터 로드됨")
    
    # TextGrid 파일 수집
    textgrid_base = Path(textgrid_dir)
    textgrid_dirs = list(textgrid_base.iterdir())
    
    # 처리할 파일 목록 생성
    tasks = []
    for tg_dir in textgrid_dirs:
        if not tg_dir.is_dir():
            continue
        
        utterance_id = tg_dir.name
        
        # TextGrid 파일 찾기
        textgrid_files = list(tg_dir.glob("*.TextGrid"))
        if not textgrid_files:
            continue
        
        textgrid_path = str(textgrid_files[0])
        
        # 메타데이터 조회
        metadata = metadata_extractor.get_metadata(utterance_id)
        if metadata is None:
            # ID 매핑 시도 (다양한 패턴)
            continue
        
        tasks.append((textgrid_path, utterance_id, metadata))
    
    logger.info(f"총 {len(tasks)}개 TextGrid 파일 처리 예정")
    
    # 병렬 처리
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_textgrid_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="TextGrid 처리 중"):
            result = future.result()
            if result:
                results.append(result)
    
    logger.info(f"총 {len(results)}개 결과 생성됨")
    
    # CSV 저장
    if results:
        fieldnames = list(results[0].keys())
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"CSV 저장 완료: {output_csv}")
    else:
        logger.warning("생성된 결과가 없습니다.")


def main():
    parser = argparse.ArgumentParser(
        description='원본 JSON과 TextGrid에서 분석용 CSV 생성'
    )
    parser.add_argument(
        '--json-dir',
        type=str,
        default='data/style/style',
        help='원본 JSON 파일 디렉토리'
    )
    parser.add_argument(
        '--textgrid-dir',
        type=str,
        default='out/results',
        help='처리된 TextGrid 파일 디렉토리'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_output/pitch_analysis_data.csv',
        help='출력 CSV 파일 경로'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='병렬 처리 워커 수'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    build_analysis_csv(
        json_dir=args.json_dir,
        textgrid_dir=args.textgrid_dir,
        output_csv=args.output,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
