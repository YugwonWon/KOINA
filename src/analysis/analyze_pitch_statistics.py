#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pitch_statistics.py

CSV 데이터를 기반으로 감정별 피치 구간 차이에 대한 통계 분석 수행

통계 방법론:
1. 일원분산분석 (One-way ANOVA): 3개 이상 감정 그룹 간 구간별 피치 차이 검정
2. Kruskal-Wallis H-test: 비모수적 대안 (정규성 가정 불충족 시)
3. 사후검정 (Post-hoc): Tukey HSD, Bonferroni, Dunn's test
4. 효과 크기: Cohen's d, Eta-squared

확장 가능한 구조:
- 새로운 분석 함수를 쉽게 추가 가능
- 다양한 그룹 변수 지원 (감정, 스타일, 성별 등)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import logging

# 통계 라이브러리
from scipy import stats
from scipy.stats import (
    shapiro, levene, f_oneway, kruskal, 
    mannwhitneyu, ttest_ind, spearmanr, pearsonr
)

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import AnovaRM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. Some post-hoc tests unavailable.")

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# 시각화 라이브러리
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버 환경용
import seaborn as sns

# ============================================================================
# 논문용 시각화 설정 (2단 구성 논문 기준)
# ============================================================================
# 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 폰트 크기 설정 (논문 2단 구성용 - 인쇄 시 가독성 확보)
plt.rcParams['font.size'] = 28
plt.rcParams['axes.titlesize'] = 34
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 26
plt.rcParams['figure.titlesize'] = 38
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['lines.markersize'] = 10

# 통일된 감정 색상 팔레트 (논문용)
# 감정에 맞는 직관적인 색상 선택
EMOTION_COLORS = {
    '무감정': '#808080',   # 회색 (중립)
    '기쁨': '#FFB347',     # 밝은 주황색 (따뜻함, 기쁨)
    '슬픔': '#6B8EAF',     # 부드러운 파랑 (차분함, 슬픔)
    '분노': '#D9534F',     # 빨강 (강렬함, 분노)
    # 추가 감정 대비
    'neutral': '#808080',
    'happy': '#FFB347',
    'sad': '#6B8EAF',
    'angry': '#D9534F',
}

# 감정 순서 (그래프 일관성용)
EMOTION_ORDER = ['무감정', '기쁨', '슬픔', '분노']

# 감정별 마커 모양 (색맹 대응 - 색상+모양으로 이중 구분)
EMOTION_MARKERS = {
    '무감정': 's',     # 정사각형
    '기쁨': 'o',       # 원
    '슬픔': '^',       # 삼각형 (위)
    '분노': 'D',       # 다이아몬드
    'neutral': 's',
    'happy': 'o',
    'sad': '^',
    'angry': 'D',
}

# 감정별 라인 스타일 (추가 구분)
EMOTION_LINESTYLES = {
    '무감정': '-',      # 실선
    '기쁨': '--',       # 점선
    '슬픔': '-.',       # 점쇄선
    '분노': ':',        # 도트선
    'neutral': '-',
    'happy': '--',
    'sad': '-.',
    'angry': ':',
}

# 변수명을 읽기 쉬운 형태로 변환하는 매핑
FEATURE_DISPLAY_NAMES = {
    # 기본 통계
    'pitch_mean': 'Pitch Mean',
    'pitch_std': 'Pitch SD',
    'pitch_min': 'Pitch Min',
    'pitch_max': 'Pitch Max',
    'pitch_range': 'Pitch Range',
    'pitch_point_count': 'Pitch Point Count',
    # 기울기 관련
    'pitch_slope': 'Overall Slope',
    'pitch_slope_abs': 'Absolute Slope',
    'pitch_slope_first_half': 'First-half Slope',
    'pitch_slope_second_half': 'Second-half Slope',
    'pitch_slope_change': 'Slope Change',
    'pitch_slope_onset': 'Onset Slope',
    'pitch_slope_offset': 'Offset Slope',
    'pitch_slope_mid': 'Mid Slope',
    # 동적 특성
    'pitch_velocity_mean': 'Velocity Mean',
    'pitch_velocity_std': 'Velocity SD',
    'pitch_velocity_abs_mean': 'Velocity Abs Mean',
    'pitch_acceleration_mean': 'Acceleration Mean',
    'pitch_inflection_count': 'Inflection Count',
    # 위치 관련
    'pitch_peak_position': 'Peak Position',
    'pitch_valley_position': 'Valley Position',
    # 구간 피치
    'pitch_bin_0_10': '0-10%',
    'pitch_bin_10_20': '10-20%',
    'pitch_bin_20_30': '20-30%',
    'pitch_bin_30_40': '30-40%',
    'pitch_bin_40_50': '40-50%',
    'pitch_bin_50_60': '50-60%',
    'pitch_bin_60_70': '60-70%',
    'pitch_bin_70_80': '70-80%',
    'pitch_bin_80_90': '80-90%',
    'pitch_bin_90_100': '90-100%',
}

def get_display_name(feature_name: str) -> str:
    """변수명을 논문용 표시 이름으로 변환"""
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]
    # 매핑에 없으면 언더스코어를 공백으로 변환
    return feature_name.replace('_', ' ').title()

def get_emotion_color(emotion: str) -> str:
    """감정에 해당하는 색상 반환"""
    return EMOTION_COLORS.get(emotion, '#808080')

def get_emotion_marker(emotion: str) -> str:
    """감정에 해당하는 마커 모양 반환"""
    return EMOTION_MARKERS.get(emotion, 'o')

def get_emotion_linestyle(emotion: str) -> str:
    """감정에 해당하는 라인 스타일 반환"""
    return EMOTION_LINESTYLES.get(emotion, '-')

def get_emotion_palette(emotions: list) -> list:
    """감정 목록에 대한 색상 팔레트 반환"""
    return [get_emotion_color(e) for e in emotions]

def get_ordered_emotions(emotions: list) -> list:
    """정의된 순서대로 감정 정렬"""
    ordered = [e for e in EMOTION_ORDER if e in emotions]
    # 정의되지 않은 감정은 뒤에 추가
    for e in emotions:
        if e not in ordered:
            ordered.append(e)
    return ordered

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 분석 결과 데이터 클래스
# ============================================================================

@dataclass
class StatisticalTestResult:
    """통계 검정 결과"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_interpretation: str = ""
    is_significant: bool = False
    alpha: float = 0.05
    additional_info: Dict = None
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.alpha
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class BinComparisonResult:
    """구간별 비교 결과"""
    bin_name: str
    omnibus_test: StatisticalTestResult
    posthoc_results: Optional[pd.DataFrame] = None
    descriptive_stats: Optional[pd.DataFrame] = None


# ============================================================================
# 통계 검정 함수들
# ============================================================================

def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Shapiro-Wilk 정규성 검정"""
    if len(data) < 3:
        return False, 1.0
    
    # 샘플이 너무 크면 일부만 사용 (Shapiro 제한)
    sample_data = data[:5000] if len(data) > 5000 else data
    
    try:
        stat, p_value = shapiro(sample_data)
        return p_value > alpha, p_value
    except Exception:
        return False, 1.0


def check_homogeneity(groups: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, float]:
    """Levene 등분산성 검정"""
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return False, 1.0
    
    try:
        stat, p_value = levene(*valid_groups)
        return p_value > alpha, p_value
    except Exception:
        return False, 1.0


def calculate_eta_squared(groups: List[np.ndarray], f_stat: float) -> float:
    """ANOVA에서 eta-squared 효과 크기 계산"""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in all_data)
    
    if ss_total == 0:
        return 0.0
    
    return ss_between / ss_total


def interpret_eta_squared(eta_sq: float) -> str:
    """Eta-squared 해석"""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d 효과 크기 계산"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_anova(groups: List[np.ndarray], group_labels: List[str]) -> StatisticalTestResult:
    """일원분산분석 실행"""
    valid_groups = [(g, l) for g, l in zip(groups, group_labels) if len(g) >= 2]
    
    if len(valid_groups) < 2:
        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=np.nan,
            p_value=1.0,
            additional_info={"error": "Insufficient groups"}
        )
    
    groups_filtered = [g for g, _ in valid_groups]
    
    try:
        f_stat, p_value = f_oneway(*groups_filtered)
        eta_sq = calculate_eta_squared(groups_filtered, f_stat)
        
        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            effect_size=eta_sq,
            effect_size_interpretation=interpret_eta_squared(eta_sq),
            additional_info={
                "n_groups": len(groups_filtered),
                "group_sizes": [len(g) for g in groups_filtered]
            }
        )
    except Exception as e:
        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=np.nan,
            p_value=1.0,
            additional_info={"error": str(e)}
        )


def run_kruskal_wallis(groups: List[np.ndarray], group_labels: List[str]) -> StatisticalTestResult:
    """Kruskal-Wallis H-test (비모수적 대안)"""
    valid_groups = [(g, l) for g, l in zip(groups, group_labels) if len(g) >= 2]
    
    if len(valid_groups) < 2:
        return StatisticalTestResult(
            test_name="Kruskal-Wallis H-test",
            statistic=np.nan,
            p_value=1.0,
            additional_info={"error": "Insufficient groups"}
        )
    
    groups_filtered = [g for g, _ in valid_groups]
    
    try:
        h_stat, p_value = kruskal(*groups_filtered)
        
        # Epsilon-squared 효과 크기 (비모수적)
        n = sum(len(g) for g in groups_filtered)
        epsilon_sq = h_stat / (n - 1) if n > 1 else 0
        
        return StatisticalTestResult(
            test_name="Kruskal-Wallis H-test",
            statistic=h_stat,
            p_value=p_value,
            effect_size=epsilon_sq,
            effect_size_interpretation="epsilon-squared",
            additional_info={
                "n_groups": len(groups_filtered),
                "group_sizes": [len(g) for g in groups_filtered]
            }
        )
    except Exception as e:
        return StatisticalTestResult(
            test_name="Kruskal-Wallis H-test",
            statistic=np.nan,
            p_value=1.0,
            additional_info={"error": str(e)}
        )


def run_tukey_hsd(df: pd.DataFrame, value_col: str, group_col: str) -> Optional[pd.DataFrame]:
    """Tukey HSD 사후검정"""
    if not HAS_STATSMODELS:
        return None
    
    try:
        tukey = pairwise_tukeyhsd(df[value_col], df[group_col], alpha=0.05)
        
        # 결과를 DataFrame으로 변환
        results = pd.DataFrame({
            'group1': tukey._results_table.data[1:, 0],
            'group2': tukey._results_table.data[1:, 1],
            'meandiff': tukey._results_table.data[1:, 2],
            'p_adj': tukey._results_table.data[1:, 3],
            'lower': tukey._results_table.data[1:, 4],
            'upper': tukey._results_table.data[1:, 5],
            'reject': tukey._results_table.data[1:, 6]
        })
        return results
    except Exception as e:
        logger.warning(f"Tukey HSD failed: {e}")
        return None


def run_dunn_test(df: pd.DataFrame, value_col: str, group_col: str) -> Optional[pd.DataFrame]:
    """Dunn's test (Kruskal-Wallis 사후검정)"""
    if not HAS_POSTHOCS:
        return None
    
    try:
        result = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
        return result
    except Exception as e:
        logger.warning(f"Dunn's test failed: {e}")
        return None


# ============================================================================
# 분석 클래스
# ============================================================================

class PitchAnalyzer:
    """피치 데이터 통계 분석기"""
    
    # 10% 구간 컬럼명
    BIN_COLUMNS = [
        'pitch_bin_0_10', 'pitch_bin_10_20', 'pitch_bin_20_30',
        'pitch_bin_30_40', 'pitch_bin_40_50', 'pitch_bin_50_60',
        'pitch_bin_60_70', 'pitch_bin_70_80', 'pitch_bin_80_90',
        'pitch_bin_90_100'
    ]
    
    # 추가 피쳐 컬럼명 (기울기 및 동적 특성)
    SLOPE_COLUMNS = [
        'pitch_slope', 'pitch_slope_abs', 'pitch_slope_first_half',
        'pitch_slope_second_half', 'pitch_slope_change'
    ]
    
    DYNAMICS_COLUMNS = [
        'pitch_velocity_mean', 'pitch_velocity_std', 'pitch_velocity_abs_mean',
        'pitch_acceleration_mean', 'pitch_inflection_count'
    ]
    
    POSITION_COLUMNS = [
        'pitch_peak_position', 'pitch_valley_position'
    ]
    
    BASIC_STATS_COLUMNS = [
        'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range', 'pitch_point_count'
    ]
    
    # 모든 분석 대상 피쳐
    ALL_FEATURE_COLUMNS = BIN_COLUMNS + SLOPE_COLUMNS + DYNAMICS_COLUMNS + POSITION_COLUMNS + BASIC_STATS_COLUMNS
    
    def __init__(self, csv_path: str):
        """CSV 파일 로드"""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df_filtered = None  # 이상치 제거된 데이터
        self.results = {}
        
        logger.info(f"데이터 로드 완료: {len(self.df)}개 레코드")
        logger.info(f"감정 분포:\n{self.df['emotion'].value_counts()}")
    
    def filter_valid_data(self, min_pitch_points: int = 3) -> pd.DataFrame:
        """유효한 데이터만 필터링"""
        if self.df_filtered is not None:
            return self.df_filtered
        
        df = self.df[self.df['pitch_point_count'] >= min_pitch_points].copy()
        
        # 0인 구간 제외 (피치 포인트가 해당 구간에 없는 경우)
        for col in self.BIN_COLUMNS:
            if col in df.columns:
                df = df[df[col] > 0]
        
        logger.info(f"필터링 후 데이터: {len(df)}개 레코드")
        return df
    
    def remove_outliers(
        self, 
        method: str = 'iqr',
        columns: List[str] = None,
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        gender_specific: bool = True
    ) -> pd.DataFrame:
        """
        이상치 제거
        
        Args:
            method: 'iqr' (IQR 방법) 또는 'zscore' (Z-score 방법)
            columns: 이상치 검사할 컬럼 (기본: pitch_mean)
            iqr_multiplier: IQR 배수 (기본: 1.5)
            z_threshold: Z-score 임계값 (기본: 3.0)
            gender_specific: 성별별로 이상치 계산 (기본: True)
        """
        df = self.filter_valid_data()
        original_count = len(df)
        
        if columns is None:
            columns = ['pitch_mean']
        
        def remove_outliers_single(data: pd.DataFrame, col: str) -> pd.DataFrame:
            if method == 'iqr':
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                return data[(data[col] >= lower) & (data[col] <= upper)]
            elif method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                if std == 0:
                    return data
                z_scores = np.abs((data[col] - mean) / std)
                return data[z_scores < z_threshold]
            else:
                return data
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if gender_specific and 'speaker_gender' in df.columns:
                # 성별별로 이상치 제거
                filtered_dfs = []
                for gender in df['speaker_gender'].unique():
                    gender_df = df[df['speaker_gender'] == gender]
                    filtered_dfs.append(remove_outliers_single(gender_df, col))
                df = pd.concat(filtered_dfs, ignore_index=True)
            else:
                df = remove_outliers_single(df, col)
        
        removed_count = original_count - len(df)
        logger.info(f"이상치 제거: {removed_count}개 ({100*removed_count/original_count:.2f}%)")
        logger.info(f"남은 데이터: {len(df)}개")
        
        self.df_filtered = df
        return df
    
    def print_descriptive_statistics(
        self, 
        group_col: str = 'emotion',
        features: List[str] = None,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        추출된 피쳐들의 기술통계량 출력
        
        Args:
            group_col: 그룹 변수
            features: 출력할 피쳐 목록 (None이면 모든 피쳐)
            save_path: CSV 저장 경로
        """
        df = self.df_filtered if self.df_filtered is not None else self.filter_valid_data()
        
        if features is None:
            features = [col for col in self.ALL_FEATURE_COLUMNS if col in df.columns]
        
        print("\n" + "=" * 100)
        print("📊 추출된 피쳐 기술통계량")
        print("=" * 100)
        
        all_stats = []
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # 전체 통계
            overall_stats = df[feature].describe()
            
            # 그룹별 통계
            group_stats = df.groupby(group_col)[feature].agg([
                'count', 'mean', 'std', 'min', 
                ('q25', lambda x: x.quantile(0.25)),
                'median',
                ('q75', lambda x: x.quantile(0.75)),
                'max'
            ]).round(4)
            
            print(f"\n▶ {feature}")
            print(f"  전체: mean={overall_stats['mean']:.4f}, std={overall_stats['std']:.4f}, " +
                  f"min={overall_stats['min']:.4f}, max={overall_stats['max']:.4f}")
            print(f"  그룹별 평균:")
            for idx, row in group_stats.iterrows():
                print(f"    {idx}: mean={row['mean']:.4f} (±{row['std']:.4f}), n={int(row['count'])}")
            
            # 저장용 데이터 추가
            for idx, row in group_stats.iterrows():
                all_stats.append({
                    'feature': feature,
                    'group': idx,
                    **row.to_dict()
                })
        
        stats_df = pd.DataFrame(all_stats)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(save_path, index=False)
            logger.info(f"기술통계량 저장: {save_path}")
        
        print("=" * 100 + "\n")
        
        return stats_df
    
    def analyze_emotion_by_bin(
        self, 
        group_col: str = 'emotion',
        alpha: float = 0.05,
        use_nonparametric: bool = True
    ) -> Dict[str, BinComparisonResult]:
        """감정별 각 구간의 피치 차이 분석"""
        
        df = self.filter_valid_data()
        
        if df.empty:
            logger.warning("분석할 데이터가 없습니다.")
            return {}
        
        results = {}
        group_labels = df[group_col].unique().tolist()
        
        for bin_col in self.BIN_COLUMNS:
            logger.info(f"분석 중: {bin_col}")
            
            # 그룹별 데이터 추출
            groups = []
            valid_labels = []
            for label in group_labels:
                group_data = df[df[group_col] == label][bin_col].dropna().values
                if len(group_data) >= 3:
                    groups.append(group_data)
                    valid_labels.append(label)
            
            if len(groups) < 2:
                logger.warning(f"{bin_col}: 충분한 그룹이 없음")
                continue
            
            # 정규성 및 등분산성 검정
            normality_results = [check_normality(g) for g in groups]
            all_normal = all(n[0] for n in normality_results)
            homogeneity, homog_p = check_homogeneity(groups)
            
            # 기술통계
            desc_stats = df.groupby(group_col)[bin_col].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
            
            # 주 검정 선택 및 실행
            if all_normal and homogeneity and not use_nonparametric:
                omnibus_test = run_anova(groups, valid_labels)
            else:
                omnibus_test = run_kruskal_wallis(groups, valid_labels)
            
            # 유의할 경우 사후검정
            posthoc_df = None
            if omnibus_test.is_significant:
                subset_df = df[df[group_col].isin(valid_labels)][[group_col, bin_col]].dropna()
                
                if omnibus_test.test_name == "One-way ANOVA":
                    posthoc_df = run_tukey_hsd(subset_df, bin_col, group_col)
                else:
                    posthoc_df = run_dunn_test(subset_df, bin_col, group_col)
            
            results[bin_col] = BinComparisonResult(
                bin_name=bin_col,
                omnibus_test=omnibus_test,
                posthoc_results=posthoc_df,
                descriptive_stats=desc_stats
            )
        
        self.results['emotion_by_bin'] = results
        return results
    
    def analyze_features_by_group(
        self, 
        features: List[str] = None,
        group_col: str = 'emotion',
        alpha: float = 0.05,
        use_nonparametric: bool = True
    ) -> Dict[str, StatisticalTestResult]:
        """
        새 피쳐들(기울기, 속도 등)에 대한 그룹별 통계 분석
        
        Args:
            features: 분석할 피쳐 목록
            group_col: 그룹 변수
            alpha: 유의수준
            use_nonparametric: 비모수적 검정 사용 여부
        """
        df = self.df_filtered if self.df_filtered is not None else self.filter_valid_data()
        
        if features is None:
            features = self.SLOPE_COLUMNS + self.DYNAMICS_COLUMNS + self.POSITION_COLUMNS + self.BASIC_STATS_COLUMNS
        
        results = {}
        group_labels = df[group_col].unique().tolist()
        
        print("\n" + "=" * 100)
        print(f"📈 피쳐별 {group_col} 그룹 간 통계 분석")
        print("=" * 100)
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            # 그룹별 데이터 추출
            groups = []
            valid_labels = []
            for label in group_labels:
                group_data = df[df[group_col] == label][feature].dropna().values
                if len(group_data) >= 3:
                    groups.append(group_data)
                    valid_labels.append(label)
            
            if len(groups) < 2:
                continue
            
            # 검정 실행
            if use_nonparametric:
                test_result = run_kruskal_wallis(groups, valid_labels)
            else:
                test_result = run_anova(groups, valid_labels)
            
            results[feature] = test_result
            
            # 결과 출력
            sig_mark = '***' if test_result.p_value < 0.001 else ('**' if test_result.p_value < 0.01 else ('*' if test_result.p_value < 0.05 else ''))
            print(f"\n▶ {feature}")
            print(f"  검정: {test_result.test_name}")
            print(f"  통계량: {test_result.statistic:.4f}, p-value: {test_result.p_value:.2e} {sig_mark}")
            if test_result.effect_size:
                print(f"  효과 크기: {test_result.effect_size:.4f}")
            
            # 그룹별 평균 출력
            group_means = df.groupby(group_col)[feature].mean()
            print(f"  그룹별 평균: " + ", ".join([f"{k}={v:.4f}" for k, v in group_means.items()]))
        
        self.results['features_by_group'] = results
        print("=" * 100 + "\n")
        
        return results
    
    def generate_feature_summary_table(self) -> pd.DataFrame:
        """피쳐 분석 결과 요약 테이블 생성"""
        if 'features_by_group' not in self.results:
            return pd.DataFrame()
        
        rows = []
        for feature, test in self.results['features_by_group'].items():
            rows.append({
                'Feature': feature,
                'Test': test.test_name,
                'Statistic': test.statistic,
                'p-value': test.p_value,
                'Significant': '***' if test.p_value < 0.001 else ('**' if test.p_value < 0.01 else ('*' if test.p_value < 0.05 else '')),
                'Effect Size': test.effect_size,
            })
        
        return pd.DataFrame(rows)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """결과 요약 테이블 생성"""
        if 'emotion_by_bin' not in self.results:
            return pd.DataFrame()
        
        rows = []
        for bin_name, result in self.results['emotion_by_bin'].items():
            test = result.omnibus_test
            rows.append({
                'Bin': bin_name,
                'Test': test.test_name,
                'Statistic': test.statistic,
                'p-value': test.p_value,
                'Significant': '***' if test.p_value < 0.001 else ('**' if test.p_value < 0.01 else ('*' if test.p_value < 0.05 else '')),
                'Effect Size': test.effect_size,
                'Effect Interpretation': test.effect_size_interpretation
            })
        
        return pd.DataFrame(rows)
    
    def plot_emotion_pitch_profile(self, output_dir: str, subset_name: str = ""):
        """감정별 피치 프로파일 시각화 (논문용 스타일)"""
        df = self.filter_valid_data()
        
        if df.empty:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 제목에 표시할 서브셋 정보
        title_suffix = f" - {subset_name}" if subset_name else ""
        
        # 1. 구간별 평균 피치 라인 플롯
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 정의된 순서대로 감정 정렬
        emotions = get_ordered_emotions(list(df['emotion'].unique()))
        # % 중복 제거한 구간 라벨
        bin_labels = [f"{i*10}-{(i+1)*10}" for i in range(10)]
        
        for emotion in emotions:
            emotion_data = df[df['emotion'] == emotion]
            means = [emotion_data[col].mean() for col in self.BIN_COLUMNS]
            stds = [emotion_data[col].std() for col in self.BIN_COLUMNS]
            
            x = range(10)
            color = get_emotion_color(emotion)
            marker = get_emotion_marker(emotion)
            linestyle = get_emotion_linestyle(emotion)
            ax.plot(x, means, marker=marker, label=emotion, linewidth=3.5, 
                   color=color, markersize=14, linestyle=linestyle)
            ax.fill_between(x, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)
        
        ax.set_xticks(range(10))
        ax.set_xticklabels(bin_labels, fontsize=24)
        ax.set_xlabel('발화 구간 (%)', fontsize=28, fontweight='bold')
        ax.set_ylabel('평균 피치 (Hz)', fontsize=28, fontweight='bold')
        # ax.set_title(f'감정별 피치 변화 양상{title_suffix}', fontsize=20, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=22, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=24)
        
        plt.tight_layout()
        plt.savefig(output_path / 'emotion_pitch_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"시각화 저장: {output_path / 'emotion_pitch_profile.png'}")
        
        # 2. 박스플롯 (각 구간별) - 논문용 스타일
        fig, axes = plt.subplots(2, 5, figsize=(24, 12))
        axes = axes.flatten()
        
        # 감정 색상 팔레트
        ordered_emotions = get_ordered_emotions(list(df['emotion'].unique()))
        palette = {e: get_emotion_color(e) for e in ordered_emotions}
        
        for idx, bin_col in enumerate(self.BIN_COLUMNS):
            ax = axes[idx]
            bin_data = df[[bin_col, 'emotion']].dropna()
            bin_data = bin_data[bin_data[bin_col] > 0]
            
            if not bin_data.empty:
                sns.boxplot(data=bin_data, x='emotion', y=bin_col, ax=ax, 
                           order=ordered_emotions, palette=palette, hue='emotion', legend=False)
                ax.set_title(f'{bin_labels[idx]}%', fontsize=22, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('피치 (Hz)', fontsize=20)
                ax.tick_params(axis='x', rotation=45, labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
        
        # plt.suptitle(f'구간별 감정에 따른 피치 분포{title_suffix}', fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'emotion_boxplot_by_bin.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"시각화 저장: {output_path / 'emotion_boxplot_by_bin.png'}")
        
        # 3. 히트맵 (p-value) - 논문용 스타일
        if 'emotion_by_bin' in self.results:
            summary = self.generate_summary_table()
            if not summary.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # p-value 시각화
                p_values = summary.set_index('Bin')['p-value']
                log_p = -np.log10(p_values.values.astype(float) + 1e-300)
                
                # 유의수준에 따른 색상
                bar_colors = ['#2ecc71' if p < 0.001 else '#f39c12' if p < 0.01 
                             else '#e74c3c' if p < 0.05 else '#95a5a6' 
                             for p in p_values.values]
                
                bars = ax.barh(range(len(p_values)), log_p, color=bar_colors, edgecolor='black', linewidth=0.5)
                ax.axvline(x=-np.log10(0.05), color='#e74c3c', linestyle='--', linewidth=2, label='p=0.05')
                ax.axvline(x=-np.log10(0.01), color='#f39c12', linestyle='--', linewidth=2, label='p=0.01')
                ax.axvline(x=-np.log10(0.001), color='#2ecc71', linestyle='--', linewidth=2, label='p=0.001')
                
                ax.set_yticks(range(len(p_values)))
                ax.set_yticklabels(p_values.index, fontsize=24)
                ax.set_xlabel('-log10(p-value)', fontsize=28, fontweight='bold')
                # ax.set_title(f'구간별 감정 효과 유의성{title_suffix}', fontsize=18, fontweight='bold')
                ax.legend(fontsize=22, loc='lower right')
                ax.tick_params(axis='x', labelsize=24)
                
                plt.tight_layout()
                plt.savefig(output_path / 'significance_barplot.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"시각화 저장: {output_path / 'significance_barplot.png'}")
    
    def plot_all_features(self, output_dir: str, group_col: str = 'emotion', subset_name: str = ""):
        """모든 피쳐에 대한 종합 시각화 (논문용 스타일)"""
        df = self.df_filtered if self.df_filtered is not None else self.filter_valid_data()
        
        if df.empty:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 제목에 표시할 서브셋 정보
        title_suffix = f" - {subset_name}" if subset_name else ""
        
        # 통일된 감정 색상 팔레트 사용
        ordered_emotions = get_ordered_emotions(list(df[group_col].unique()))
        color_map = {e: get_emotion_color(e) for e in ordered_emotions}
        
        # =====================================================================
        # 1. 기울기 피쳐 비교 (바 차트)
        # =====================================================================
        slope_features = [col for col in self.SLOPE_COLUMNS if col in df.columns]
        if slope_features:
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            
            for idx, feature in enumerate(slope_features):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                
                # 감정 순서대로 정렬
                means = df.groupby(group_col)[feature].mean().reindex(ordered_emotions)
                stds = df.groupby(group_col)[feature].std().reindex(ordered_emotions)
                
                bars = ax.bar(means.index, means.values, 
                             color=[color_map[e] for e in means.index],
                             yerr=stds.values, capsize=5, alpha=0.85,
                             edgecolor='black', linewidth=0.5)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                display_name = get_display_name(feature)
                ax.set_title(f'{display_name}', fontsize=28, fontweight='bold')
                ax.set_ylabel('', fontsize=26)
                ax.tick_params(axis='x', rotation=45, labelsize=24)
                ax.tick_params(axis='y', labelsize=24)
                
                # 유의성 표시
                if 'features_by_group' in self.results and feature in self.results['features_by_group']:
                    p_val = self.results['features_by_group'][feature].p_value
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                    ax.set_title(f'{display_name} ({sig})', fontsize=28, fontweight='bold')
            
            # 빈 축 숨기기
            for idx in range(len(slope_features), len(axes)):
                axes[idx].set_visible(False)
            
            # plt.suptitle(f'감정별 피치 기울기 피쳐 비교{title_suffix}', fontsize=28, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'slope_features_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'slope_features_comparison.png'}")
        
        # =====================================================================
        # 2. 동적 특성 피쳐 비교 (바 차트)
        # =====================================================================
        dynamics_features = [col for col in self.DYNAMICS_COLUMNS if col in df.columns]
        if dynamics_features:
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            
            for idx, feature in enumerate(dynamics_features):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                
                # 감정 순서대로 정렬
                means = df.groupby(group_col)[feature].mean().reindex(ordered_emotions)
                stds = df.groupby(group_col)[feature].std().reindex(ordered_emotions)
                
                bars = ax.bar(means.index, means.values,
                             color=[color_map[e] for e in means.index],
                             yerr=stds.values, capsize=5, alpha=0.85,
                             edgecolor='black', linewidth=0.5)
                
                display_name = get_display_name(feature)
                ax.set_title(f'{display_name}', fontsize=28, fontweight='bold')
                ax.set_ylabel('', fontsize=26)
                ax.tick_params(axis='x', rotation=45, labelsize=24)
                ax.tick_params(axis='y', labelsize=24)
                
                if 'features_by_group' in self.results and feature in self.results['features_by_group']:
                    p_val = self.results['features_by_group'][feature].p_value
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                    ax.set_title(f'{display_name} ({sig})', fontsize=28, fontweight='bold')
            
            for idx in range(len(dynamics_features), len(axes)):
                axes[idx].set_visible(False)
            
            # plt.suptitle(f'감정별 피치 동적 특성 비교{title_suffix}', fontsize=28, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'dynamics_features_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'dynamics_features_comparison.png'}")
        
        # =====================================================================
        # 3. 피크/밸리 위치 비교 (박스플롯) - 논문용 스타일
        # =====================================================================
        position_features = [col for col in self.POSITION_COLUMNS if col in df.columns]
        if position_features:
            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            palette = {e: get_emotion_color(e) for e in ordered_emotions}
            
            for idx, feature in enumerate(position_features):
                ax = axes[idx]
                sns.boxplot(data=df, x=group_col, y=feature, ax=ax, 
                           order=ordered_emotions, palette=palette, hue=group_col, legend=False)
                sns.stripplot(data=df.sample(min(1000, len(df))), x=group_col, y=feature, 
                             ax=ax, color='black', alpha=0.1, size=2, order=ordered_emotions)
                
                display_name = get_display_name(feature)
                ax.set_title(f'{display_name}', fontsize=30, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('발화 위치 (%)', fontsize=28)
                ax.tick_params(axis='x', rotation=45, labelsize=26)
                ax.tick_params(axis='y', labelsize=26)
                
                if 'features_by_group' in self.results and feature in self.results['features_by_group']:
                    p_val = self.results['features_by_group'][feature].p_value
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                    ax.set_title(f'{display_name} ({sig})', fontsize=30, fontweight='bold')
            
            # plt.suptitle(f'감정별 피치 피크/밸리 위치 비교{title_suffix}', fontsize=28, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'position_features_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'position_features_comparison.png'}")
        
        # =====================================================================
        # 4. 기본 통계 피쳐 (바이올린 플롯) - 논문용 스타일
        # =====================================================================
        basic_features = ['pitch_mean', 'pitch_std', 'pitch_range']
        basic_features = [col for col in basic_features if col in df.columns]
        
        if basic_features:
            fig, axes = plt.subplots(1, 3, figsize=(22, 9))
            palette = {e: get_emotion_color(e) for e in ordered_emotions}
            
            for idx, feature in enumerate(basic_features):
                ax = axes[idx]
                sns.violinplot(data=df, x=group_col, y=feature, ax=ax, 
                              order=ordered_emotions, palette=palette, inner='box', hue=group_col, legend=False)
                
                display_name = get_display_name(feature)
                ax.set_title(f'{display_name}', fontsize=30, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Hz', fontsize=28)
                ax.tick_params(axis='x', rotation=45, labelsize=26)
                ax.tick_params(axis='y', labelsize=26)
                
                if 'features_by_group' in self.results and feature in self.results['features_by_group']:
                    p_val = self.results['features_by_group'][feature].p_value
                    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                    ax.set_title(f'{display_name} ({sig})', fontsize=30, fontweight='bold')
            
            # plt.suptitle(f'감정별 기본 피치 통계 비교{title_suffix}', fontsize=28, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / 'basic_stats_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'basic_stats_comparison.png'}")
        
        # =====================================================================
        # 5. 종합 히트맵 (감정별 피쳐 평균 z-score) - 논문용 스타일
        # =====================================================================
        all_features = slope_features + dynamics_features + position_features + basic_features
        if all_features:
            # z-score 정규화 (감정 순서대로)
            feature_means = df.groupby(group_col)[all_features].mean().reindex(ordered_emotions)
            feature_zscore = (feature_means - feature_means.mean()) / feature_means.std()
            
            # 피쳐명을 읽기 쉽게 변환
            feature_zscore_display = feature_zscore.copy()
            feature_zscore_display.columns = [get_display_name(f) for f in feature_zscore.columns]
            
            fig, ax = plt.subplots(figsize=(20, 12))
            sns.heatmap(feature_zscore_display.T, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=ax, cbar_kws={'label': 'Z-score'},
                       annot_kws={'size': 20})
            
            # ax.set_title(f'감정별 피쳐 평균 (Z-score 정규화){title_suffix}', fontsize=26, fontweight='bold')
            ax.set_xlabel('감정', fontsize=28, fontweight='bold')
            ax.set_ylabel('피쳐', fontsize=28, fontweight='bold')
            ax.tick_params(axis='x', labelsize=24)
            ax.tick_params(axis='y', labelsize=20)
            
            plt.tight_layout()
            plt.savefig(output_path / 'feature_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'feature_heatmap.png'}")
        
        # =====================================================================
        # 6. 피쳐별 유의성 종합 차트 - 논문용 스타일
        # =====================================================================
        if 'features_by_group' in self.results:
            feature_pvals = {f: r.p_value for f, r in self.results['features_by_group'].items()}
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            features = list(feature_pvals.keys())
            display_names = [get_display_name(f) for f in features]
            pvals = list(feature_pvals.values())
            log_pvals = [-np.log10(p + 1e-300) for p in pvals]
            
            # 색상 지정 (유의수준에 따라)
            colors = ['#2ecc71' if p < 0.001 else '#f39c12' if p < 0.01 
                     else '#e74c3c' if p < 0.05 else '#95a5a6' for p in pvals]
            
            bars = ax.barh(range(len(features)), log_pvals, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=0.5)
            
            # 유의수준 선
            ax.axvline(x=-np.log10(0.05), color='#e74c3c', linestyle='--', linewidth=2, label='p=0.05')
            ax.axvline(x=-np.log10(0.01), color='#f39c12', linestyle='--', linewidth=2, label='p=0.01')
            ax.axvline(x=-np.log10(0.001), color='#2ecc71', linestyle='--', linewidth=2, label='p=0.001')
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(display_names, fontsize=24)
            ax.set_xlabel('-log10(p-value)', fontsize=28, fontweight='bold')
            # ax.set_title(f'피쳐별 감정 효과 유의성{title_suffix}', fontsize=22, fontweight='bold')
            ax.legend(loc='lower right', fontsize=22)
            ax.tick_params(axis='x', labelsize=24)
            
            plt.tight_layout()
            plt.savefig(output_path / 'feature_significance.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'feature_significance.png'}")
        
        # =====================================================================
        # 7. 감정별 레이더 차트 (주요 피쳐) - 논문용 스타일
        # =====================================================================
        radar_features = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope', 
                         'pitch_velocity_abs_mean', 'pitch_inflection_count']
        radar_features = [f for f in radar_features if f in df.columns]
        
        if len(radar_features) >= 3:
            # 정규화 (0-1 스케일) - 감정 순서대로
            feature_data = df.groupby(group_col)[radar_features].mean().reindex(ordered_emotions)
            feature_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            
            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
            angles += angles[:1]  # 닫힌 도형
            
            radar_display_names = [get_display_name(f) for f in radar_features]
            
            for emotion in ordered_emotions:
                values = feature_norm.loc[emotion].values.tolist()
                values += values[:1]
                color = get_emotion_color(emotion)
                ax.plot(angles, values, 'o-', linewidth=2.5, label=emotion, color=color, markersize=8)
                ax.fill(angles, values, alpha=0.15, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_display_names, fontsize=24)
            # ax.set_title(f'감정별 피쳐 프로파일 (정규화){title_suffix}', fontsize=22, fontweight='bold', pad=25)
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=22)
            ax.tick_params(axis='y', labelsize=20)
            
            plt.tight_layout()
            plt.savefig(output_path / 'emotion_radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {output_path / 'emotion_radar_chart.png'}")
        
        # =====================================================================
        # 8. 상관관계 분석 추가
        # =====================================================================
        self.analyze_correlations(output_path, group_col)

    def analyze_correlations(self, output_path: Path, group_col: str = 'emotion'):
        """
        피쳐 간 상관관계 분석 (논문용 스타일)
        
        1. 전체 피쳐 간 상관관계 히트맵
        2. 감정별 상관관계 비교
        3. 주요 상관관계 요약 테이블
        """
        df = self.df_filtered if self.df_filtered is not None else self.filter_valid_data()
        output_path = Path(output_path)
        
        # 분석할 피쳐들
        analysis_features = [col for col in self.ALL_FEATURE_COLUMNS if col in df.columns]
        
        # 피쳐 그룹 정의 (시각화용)
        feature_groups = {
            'Bin Statistics': [c for c in self.BIN_COLUMNS if c in df.columns],
            'Slope Features': [c for c in self.SLOPE_COLUMNS if c in df.columns],
            'Dynamics Features': [c for c in self.DYNAMICS_COLUMNS if c in df.columns],
            'Position Features': [c for c in self.POSITION_COLUMNS if c in df.columns],
            'Basic Statistics': [c for c in self.BASIC_STATS_COLUMNS if c in df.columns]
        }
        
        # 주요 피쳐 (비구간 피쳐)
        key_features = (self.SLOPE_COLUMNS + self.DYNAMICS_COLUMNS + 
                       self.POSITION_COLUMNS + self.BASIC_STATS_COLUMNS)
        key_features = [f for f in key_features if f in df.columns]
        
        if len(key_features) < 2:
            logger.warning("상관관계 분석을 위한 피쳐가 부족합니다.")
            return
        
        print("\n" + "=" * 100)
        print("📊 피쳐 간 상관관계 분석")
        print("=" * 100)
        
        # =====================================================================
        # 1. 전체 피쳐 간 상관관계 (Pearson & Spearman) - 논문용 스타일
        # =====================================================================
        # Pearson 상관계수 (선형 관계)
        pearson_corr = df[key_features].corr(method='pearson')
        
        # Spearman 상관계수 (단조 관계 - 비선형에도 강건)
        spearman_corr = df[key_features].corr(method='spearman')
        
        # Pearson 상관관계 히트맵
        fig, axes = plt.subplots(2, 1, figsize=(16, 28))
        
        # 피쳐 display name 리스트
        display_labels = [get_display_name(f) for f in key_features]
        
        # 상관행렬에 display name 적용
        pearson_corr_display = pearson_corr.copy()
        pearson_corr_display.index = display_labels
        pearson_corr_display.columns = display_labels
        
        spearman_corr_display = spearman_corr.copy()
        spearman_corr_display.index = display_labels
        spearman_corr_display.columns = display_labels
        
        # Pearson
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
        sns.heatmap(pearson_corr_display, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=axes[0], cbar_kws={'label': 'Pearson r'},
                   annot_kws={'size': 16}, vmin=-1, vmax=1)
        axes[0].set_title('Pearson 상관계수 (선형 관계)', fontsize=26, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=90, labelsize=16)
        axes[0].tick_params(axis='y', rotation=0, labelsize=16)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='center')
        
        # Spearman
        sns.heatmap(spearman_corr_display, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=axes[1], cbar_kws={'label': 'Spearman ρ'},
                   annot_kws={'size': 16}, vmin=-1, vmax=1)
        axes[1].set_title('Spearman 상관계수 (단조 관계)', fontsize=26, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=90, labelsize=16)
        axes[1].tick_params(axis='y', rotation=0, labelsize=16)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), ha='center')
        
        # plt.suptitle('피쳐 간 상관관계 분석', fontsize=24, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)  # 상하 subplot 간격
        plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'correlation_matrix.png'}")
        
        # =====================================================================
        # 1-2. 개별 상관관계 히트맵 (논문용 - 더 큰 사이즈)
        # =====================================================================
        # Pearson 개별 저장
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(pearson_corr_display, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
                   annot_kws={'size': 16}, vmin=-1, vmax=1)
        # ax.set_title('Pearson 상관계수 (선형 관계)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=18)
        ax.tick_params(axis='y', rotation=0, labelsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, left=0.18)
        plt.savefig(output_path / 'correlation_pearson.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'correlation_pearson.png'}")
        
        # Spearman 개별 저장
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(spearman_corr_display, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8},
                   annot_kws={'size': 16}, vmin=-1, vmax=1)
        # ax.set_title('Spearman 상관계수 (단조 관계)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=18)
        ax.tick_params(axis='y', rotation=0, labelsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, left=0.18)
        plt.savefig(output_path / 'correlation_spearman.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'correlation_spearman.png'}")
        
        # =====================================================================
        # 2. 강한 상관관계 요약 테이블
        # =====================================================================
        strong_correlations = []
        
        for i in range(len(key_features)):
            for j in range(i + 1, len(key_features)):
                feat1, feat2 = key_features[i], key_features[j]
                pearson_r = pearson_corr.loc[feat1, feat2]
                spearman_rho = spearman_corr.loc[feat1, feat2]
                
                # 통계적 유의성 검정
                valid_data = df[[feat1, feat2]].dropna()
                if len(valid_data) > 3:
                    _, pearson_p = pearsonr(valid_data[feat1], valid_data[feat2])
                    _, spearman_p = spearmanr(valid_data[feat1], valid_data[feat2])
                else:
                    pearson_p, spearman_p = 1.0, 1.0
                
                # 상관관계 강도 해석
                abs_r = abs(pearson_r)
                if abs_r >= 0.7:
                    strength = "Strong"
                elif abs_r >= 0.4:
                    strength = "Moderate"
                elif abs_r >= 0.2:
                    strength = "Weak"
                else:
                    strength = "Negligible"
                
                strong_correlations.append({
                    'Feature 1': feat1,
                    'Feature 2': feat2,
                    'Pearson r': round(pearson_r, 4),
                    'Pearson p-value': pearson_p,
                    'Spearman ρ': round(spearman_rho, 4),
                    'Spearman p-value': spearman_p,
                    'Strength': strength,
                    'Significant (p<0.05)': 'Yes' if pearson_p < 0.05 else 'No'
                })
        
        corr_df = pd.DataFrame(strong_correlations)
        corr_df = corr_df.sort_values('Pearson r', key=abs, ascending=False)
        corr_df.to_csv(output_path / 'correlation_summary.csv', index=False)
        
        # 강한 상관관계만 출력
        strong_only = corr_df[corr_df['Strength'].isin(['Strong', 'Moderate'])]
        print(f"\n▶ 강한/중간 상관관계 (|r| >= 0.4): {len(strong_only)}개")
        if not strong_only.empty:
            print(strong_only[['Feature 1', 'Feature 2', 'Pearson r', 'Strength']].head(15).to_string(index=False))
        
        # =====================================================================
        # 3. 감정별 상관관계 비교 - 논문용 스타일
        # =====================================================================
        emotions = df[group_col].unique()
        ordered_emotions = get_ordered_emotions(list(emotions))
        
        # 감정별 주요 상관관계 비교
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        axes = axes.flatten()
        
        emotion_corr_summary = {}
        
        for idx, emotion in enumerate(ordered_emotions[:4]):  # 최대 4개 감정
            emotion_df = df[df[group_col] == emotion]
            emotion_corr = emotion_df[key_features].corr(method='pearson')
            emotion_corr_summary[emotion] = emotion_corr
            
            # display name 적용
            emotion_corr_display = emotion_corr.copy()
            emotion_corr_display.index = display_labels
            emotion_corr_display.columns = display_labels
            
            mask = np.triu(np.ones_like(emotion_corr, dtype=bool), k=1)
            sns.heatmap(emotion_corr_display, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=axes[idx], cbar_kws={'label': 'r'},
                       annot_kws={'size': 13}, vmin=-1, vmax=1)
            axes[idx].set_title(f'{emotion}', fontsize=26, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=90, labelsize=14)
            axes[idx].tick_params(axis='y', rotation=0, labelsize=14)
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), ha='center')
        
        # plt.suptitle('감정별 피쳐 상관관계 비교', fontsize=24, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, hspace=0.35)  # 하단 여백 및 subplot 간격 추가
        plt.savefig(output_path / 'correlation_by_emotion.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'correlation_by_emotion.png'}")
        
        # =====================================================================
        # 4. 감정 간 상관관계 차이 분석
        # =====================================================================
        if len(emotions) >= 2:
            # 감정 간 상관관계 패턴 차이
            corr_diff_data = []
            
            emotion_list = list(emotions)[:4]
            for i in range(len(key_features)):
                for j in range(i + 1, len(key_features)):
                    feat1, feat2 = key_features[i], key_features[j]
                    
                    row = {'Feature Pair': f"{feat1} vs {feat2}"}
                    for emotion in emotion_list:
                        row[emotion] = round(emotion_corr_summary[emotion].loc[feat1, feat2], 3)
                    
                    # 감정 간 상관관계 차이 (최대 - 최소)
                    corr_values = [emotion_corr_summary[e].loc[feat1, feat2] for e in emotion_list]
                    row['Range'] = round(max(corr_values) - min(corr_values), 3)
                    
                    corr_diff_data.append(row)
            
            corr_diff_df = pd.DataFrame(corr_diff_data)
            corr_diff_df = corr_diff_df.sort_values('Range', ascending=False)
            corr_diff_df.to_csv(output_path / 'correlation_diff_by_emotion.csv', index=False)
            
            # 감정 간 상관관계 차이가 큰 피쳐쌍 출력
            print(f"\n▶ 감정 간 상관관계 패턴이 다른 피쳐쌍 (Range > 0.2):")
            diff_pairs = corr_diff_df[corr_diff_df['Range'] > 0.2]
            if not diff_pairs.empty:
                print(diff_pairs.head(10).to_string(index=False))
            else:
                print("   모든 피쳐쌍이 감정 간 유사한 상관관계 패턴을 보입니다.")
        
        # =====================================================================
        # 5. 다중공선성 분석 (VIF - Variance Inflation Factor)
        # =====================================================================
        print(f"\n▶ 다중공선성 분석 (VIF)")
        print("   VIF > 10: 심각한 다중공선성, VIF > 5: 주의 필요")
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # 결측치 제거 및 상수항 추가
            vif_df = df[key_features].dropna()
            
            # VIF 계산
            vif_data = []
            for i, feature in enumerate(key_features):
                try:
                    vif = variance_inflation_factor(vif_df.values, i)
                    vif_data.append({'Feature': feature, 'VIF': round(vif, 2)})
                except:
                    vif_data.append({'Feature': feature, 'VIF': np.nan})
            
            vif_result = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
            vif_result.to_csv(output_path / 'vif_multicollinearity.csv', index=False)
            
            # 높은 VIF 경고
            high_vif = vif_result[vif_result['VIF'] > 5]
            if not high_vif.empty:
                print(f"   ⚠️ 다중공선성 주의 피쳐:")
                print(high_vif.to_string(index=False))
            else:
                print("   ✅ 모든 피쳐의 VIF < 5 (다중공선성 문제 없음)")
                
        except ImportError:
            print("   ⚠️ statsmodels가 필요합니다. VIF 분석 생략.")
        except Exception as e:
            print(f"   ⚠️ VIF 계산 중 오류: {e}")
        
        print("\n" + "=" * 100)
        print(f"📁 상관관계 분석 결과 저장: {output_path}")
        print("   - correlation_matrix.png: Pearson/Spearman 상관계수 히트맵")
        print("   - correlation_summary.csv: 모든 피쳐쌍 상관관계 요약")
        print("   - correlation_by_emotion.png: 감정별 상관관계 비교")
        print("   - correlation_diff_by_emotion.csv: 감정 간 상관관계 차이")
        print("   - vif_multicollinearity.csv: 다중공선성 분석 (VIF)")
        print("=" * 100)

    def export_results(self, output_dir: str):
        """분석 결과 내보내기"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 요약 테이블
        summary = self.generate_summary_table()
        if not summary.empty:
            summary.to_csv(output_path / 'statistical_summary.csv', index=False)
            logger.info(f"요약 테이블 저장: {output_path / 'statistical_summary.csv'}")
        
        # 상세 결과
        if 'emotion_by_bin' in self.results:
            for bin_name, result in self.results['emotion_by_bin'].items():
                bin_dir = output_path / bin_name
                bin_dir.mkdir(exist_ok=True)
                
                # 기술통계
                if result.descriptive_stats is not None:
                    result.descriptive_stats.to_csv(bin_dir / 'descriptive_stats.csv')
                
                # 사후검정 결과
                if result.posthoc_results is not None:
                    result.posthoc_results.to_csv(bin_dir / 'posthoc_results.csv')
        
        logger.info(f"상세 결과 저장 완료: {output_path}")


def generate_gender_comparison_plots(
    male_df: pd.DataFrame,
    female_df: pd.DataFrame,
    output_dir: str,
    group_col: str = 'emotion'
):
    """
    남녀 비교 subplot 그래프 생성 (논문용)
    
    Args:
        male_df: 남성 데이터 (필터링 완료된)
        female_df: 여성 데이터 (필터링 완료된)
        output_dir: 출력 디렉토리
        group_col: 그룹 변수
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 남녀 비교 시각화 생성")
    print("=" * 80)
    
    # 감정 순서 및 색상
    ordered_emotions = get_ordered_emotions(list(male_df[group_col].unique()))
    
    BIN_COLUMNS = [f'pitch_bin_{i*10}_{(i+1)*10}' for i in range(10)]
    bin_labels = [f"{i*10}-{(i+1)*10}" for i in range(10)]
    
    # =========================================================================
    # 1. 감정별 피치 변화 양상 (남녀 비교) - 메인 그래프
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    for ax_idx, (df, gender_name, gender_label) in enumerate([
        (male_df, '남성', 'Male'),
        (female_df, '여성', 'Female')
    ]):
        ax = axes[ax_idx]
        
        for emotion in ordered_emotions:
            emotion_data = df[df[group_col] == emotion]
            means = [emotion_data[col].mean() for col in BIN_COLUMNS if col in df.columns]
            stds = [emotion_data[col].std() for col in BIN_COLUMNS if col in df.columns]
            
            x = range(len(means))
            color = get_emotion_color(emotion)
            marker = get_emotion_marker(emotion)
            linestyle = get_emotion_linestyle(emotion)
            ax.plot(x, means, marker=marker, label=emotion, linewidth=3.5, 
                   color=color, markersize=14, linestyle=linestyle)
            ax.fill_between(x, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)
        
        ax.set_xticks(range(10))
        ax.set_xticklabels(bin_labels, fontsize=24)
        ax.set_xlabel('발화 구간 (%)', fontsize=28, fontweight='bold')
        ax.set_ylabel('평균 피치 (Hz)', fontsize=28, fontweight='bold')
        ax.set_title(f'{gender_name} ({gender_label})', fontsize=30, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=22, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=24)
    
    # plt.suptitle('감정별 피치 변화 양상', fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'gender_comparison_pitch_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"시각화 저장: {output_path / 'gender_comparison_pitch_profile.png'}")
    
    # =========================================================================
    # 2. 구간별 박스플롯 비교 (2x10 subplot)
    # =========================================================================
    fig, axes = plt.subplots(2, 10, figsize=(36, 12))
    
    palette = {e: get_emotion_color(e) for e in ordered_emotions}
    
    for row_idx, (df, gender_name) in enumerate([
        (male_df, '남성'),
        (female_df, '여성')
    ]):
        for col_idx, bin_col in enumerate(BIN_COLUMNS):
            if bin_col not in df.columns:
                continue
            ax = axes[row_idx, col_idx]
            bin_data = df[[bin_col, group_col]].dropna()
            bin_data = bin_data[bin_data[bin_col] > 0]
            
            if not bin_data.empty:
                sns.boxplot(data=bin_data, x=group_col, y=bin_col, ax=ax,
                           order=ordered_emotions, palette=palette)
                
            if row_idx == 0:
                ax.set_title(f'{bin_labels[col_idx]}%', fontsize=18, fontweight='bold')
            ax.set_xlabel('')
            if col_idx == 0:
                ax.set_ylabel(f'{gender_name}\n피치 (Hz)', fontsize=18, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=45, labelsize=14)
            ax.tick_params(axis='y', labelsize=16)
    
    # plt.suptitle('구간별 감정에 따른 피치 분포 (남녀 비교)', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'gender_comparison_boxplot_by_bin.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"시각화 저장: {output_path / 'gender_comparison_boxplot_by_bin.png'}")
    
    # =========================================================================
    # 3. 평균 피치 비교 (바이올린 플롯) - pitch_mean만 큰 그래프로
    # =========================================================================
    if 'pitch_mean' in male_df.columns and 'pitch_mean' in female_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        for ax_idx, (df, gender_name, gender_label) in enumerate([
            (male_df, '남성', 'Male'),
            (female_df, '여성', 'Female')
        ]):
            ax = axes[ax_idx]
            sns.violinplot(data=df, x=group_col, y='pitch_mean', ax=ax,
                          order=ordered_emotions, palette=palette, inner='box')
            
            ax.set_xlabel('', fontsize=28, fontweight='bold')
            ax.set_ylabel(f'{get_display_name("pitch_mean")} (Hz)', fontsize=28, fontweight='bold')
            ax.set_title(f'{gender_name} ({gender_label})', fontsize=28, fontweight='bold')
            ax.tick_params(axis='x', rotation=0, labelsize=24)
            ax.tick_params(axis='y', labelsize=24)
        
        # plt.suptitle('감정별 평균 피치 분포 비교', fontsize=26, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'gender_comparison_pitch_mean.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'gender_comparison_pitch_mean.png'}")
    
    # =========================================================================
    # 4. 기울기 피쳐 비교
    # =========================================================================
    SLOPE_COLUMNS = ['pitch_slope', 'pitch_slope_first_half', 'pitch_slope_second_half',
                     'pitch_slope_onset', 'pitch_slope_offset', 'pitch_slope_mid']
    slope_features = [f for f in SLOPE_COLUMNS if f in male_df.columns and f in female_df.columns]
    
    if slope_features:
        fig, axes = plt.subplots(2, len(slope_features), figsize=(5*len(slope_features), 12))
        
        for row_idx, (df, gender_name) in enumerate([
            (male_df, '남성'),
            (female_df, '여성')
        ]):
            for col_idx, feature in enumerate(slope_features):
                ax = axes[row_idx, col_idx] if len(slope_features) > 1 else axes[row_idx]
                
                # 감정 순서대로 정렬
                means = df.groupby(group_col)[feature].mean().reindex(ordered_emotions)
                stds = df.groupby(group_col)[feature].std().reindex(ordered_emotions)
                
                bars = ax.bar(means.index, means.values,
                             color=[palette[e] for e in means.index],
                             yerr=stds.values, capsize=5, alpha=0.85,
                             edgecolor='black', linewidth=0.5)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                display_name = get_display_name(feature)
                if row_idx == 0:
                    ax.set_title(f'{display_name}', fontsize=26, fontweight='bold')
                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(f'{gender_name}\n값', fontsize=24, fontweight='bold')
                else:
                    ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=45, labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
        
        # plt.suptitle('감정별 피치 기울기 피쳐 비교 (남녀)', fontsize=24, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'gender_comparison_slope_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'gender_comparison_slope_features.png'}")
    
    # =========================================================================
    # 5. 레이더 차트 비교
    # =========================================================================
    radar_features = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
                     'pitch_velocity_abs_mean', 'pitch_inflection_count']
    radar_features = [f for f in radar_features if f in male_df.columns and f in female_df.columns]
    
    if len(radar_features) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw=dict(projection='polar'))
        
        for ax_idx, (df, gender_name) in enumerate([
            (male_df, '남성'),
            (female_df, '여성')
        ]):
            ax = axes[ax_idx]
            
            # 정규화 (0-1 스케일)
            feature_data = df.groupby(group_col)[radar_features].mean().reindex(ordered_emotions)
            feature_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            
            angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
            angles += angles[:1]
            
            for emotion in ordered_emotions:
                values = feature_norm.loc[emotion].values.tolist()
                values += values[:1]
                color = get_emotion_color(emotion)
                marker = get_emotion_marker(emotion)
                ax.plot(angles, values, marker=marker, linestyle='-', linewidth=2.5, 
                       label=emotion, color=color, markersize=10)
                ax.fill(angles, values, alpha=0.15, color=color)
            
            radar_display_names = [get_display_name(f) for f in radar_features]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_display_names, fontsize=22)
            ax.set_title(f'{gender_name}', fontsize=26, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=22)
        
        # plt.suptitle('감정별 피쳐 프로파일 (남녀 비교)', fontsize=24, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(output_path / 'gender_comparison_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'gender_comparison_radar_chart.png'}")
    
    # =========================================================================
    # 6. 피크/밸리 위치 비교 (남녀)
    # =========================================================================
    POSITION_COLUMNS = ['pitch_peak_position', 'pitch_valley_position']
    position_features = [f for f in POSITION_COLUMNS if f in male_df.columns and f in female_df.columns]
    
    if position_features:
        fig, axes = plt.subplots(2, len(position_features), figsize=(8*len(position_features), 14))
        
        for row_idx, (df, gender_name) in enumerate([
            (male_df, '남성'),
            (female_df, '여성')
        ]):
            for col_idx, feature in enumerate(position_features):
                ax = axes[row_idx, col_idx] if len(position_features) > 1 else axes[row_idx]
                
                sns.boxplot(data=df, x=group_col, y=feature, ax=ax,
                           order=ordered_emotions, hue=group_col, palette=palette, legend=False)
                sns.stripplot(data=df.sample(min(1000, len(df))), x=group_col, y=feature,
                             ax=ax, color='black', alpha=0.1, size=2, order=ordered_emotions)
                
                display_name = get_display_name(feature)
                if row_idx == 0:
                    ax.set_title(f'{display_name}', fontsize=26, fontweight='bold')
                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(f'{gender_name}\n발화 위치 (%)', fontsize=24, fontweight='bold')
                else:
                    ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=45, labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
        
        # plt.suptitle('감정별 피치 피크/밸리 위치 비교 (남녀)', fontsize=24, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'gender_comparison_position_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'gender_comparison_position_features.png'}")
    
    # =========================================================================
    # 7. 동적 특성 비교 (남녀)
    # =========================================================================
    DYNAMICS_COLUMNS = ['pitch_velocity_mean', 'pitch_velocity_std', 'pitch_velocity_abs_mean',
                        'pitch_acceleration_mean', 'pitch_inflection_count', 'pitch_slope_abs']
    dynamics_features = [f for f in DYNAMICS_COLUMNS if f in male_df.columns and f in female_df.columns]
    
    if dynamics_features:
        n_features = len(dynamics_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(2, n_features, figsize=(5*n_features, 14))
        
        for row_idx, (df, gender_name) in enumerate([
            (male_df, '남성'),
            (female_df, '여성')
        ]):
            for col_idx, feature in enumerate(dynamics_features):
                ax = axes[row_idx, col_idx] if n_features > 1 else axes[row_idx]
                
                # 감정 순서대로 정렬
                means = df.groupby(group_col)[feature].mean().reindex(ordered_emotions)
                stds = df.groupby(group_col)[feature].std().reindex(ordered_emotions)
                
                bars = ax.bar(means.index, means.values,
                             color=[palette[e] for e in means.index],
                             yerr=stds.values, capsize=5, alpha=0.85,
                             edgecolor='black', linewidth=0.5)
                
                display_name = get_display_name(feature)
                if row_idx == 0:
                    ax.set_title(f'{display_name}', fontsize=24, fontweight='bold')
                ax.set_xlabel('')
                if col_idx == 0:
                    ax.set_ylabel(f'{gender_name}\n값', fontsize=22, fontweight='bold')
                else:
                    ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=45, labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
        
        # plt.suptitle('감정별 피치 동적 특성 비교 (남녀)', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'gender_comparison_dynamics_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {output_path / 'gender_comparison_dynamics_features.png'}")
    
    print(f"\n✅ 남녀 비교 시각화 완료!")
    print(f"📁 저장 위치: {output_path}")
    print("   - gender_comparison_pitch_profile.png: 감정별 피치 변화 양상")
    print("   - gender_comparison_boxplot_by_bin.png: 구간별 박스플롯")
    print("   - gender_comparison_pitch_mean.png: 평균 피치 분포 비교")
    print("   - gender_comparison_slope_features.png: 기울기 피쳐 비교")
    print("   - gender_comparison_position_features.png: 피크/밸리 위치 비교")
    print("   - gender_comparison_dynamics_features.png: 동적 특성 비교")
    print("   - gender_comparison_radar_chart.png: 레이더 차트")
    print("=" * 80)


def run_analysis_for_subset(
    df: pd.DataFrame,
    output_dir: str,
    group_col: str,
    alpha: float,
    use_nonparametric: bool,
    remove_outliers: bool,
    outlier_method: str,
    iqr_multiplier: float,
    analyze_all_features: bool,
    subset_name: str = "전체"
):
    """
    주어진 데이터 서브셋에 대해 전체 분석 파이프라인 실행
    
    Args:
        df: 분석할 데이터프레임
        output_dir: 출력 디렉토리
        group_col: 그룹 변수
        alpha: 유의수준
        use_nonparametric: 비모수적 검정 사용 여부
        remove_outliers: 이상치 제거 여부
        outlier_method: 이상치 제거 방법
        iqr_multiplier: IQR 배수
        analyze_all_features: 모든 피쳐 분석 여부
        subset_name: 서브셋 이름 (로깅용)
    """
    import io
    import sys
    from contextlib import redirect_stdout
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 출력을 파일과 콘솔 모두에 저장하기 위한 클래스
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # 로그 파일 열기
    log_file = open(output_path / 'analysis_log.txt', 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)
    
    try:
        print("\n" + "=" * 80)
        print(f"📊 [{subset_name}] 분석 시작")
        print(f"   데이터 수: {len(df)}개")
        print(f"   출력 경로: {output_dir}")
        print("=" * 80)
        
        # 임시 CSV 저장 후 분석기 생성
        temp_csv = output_path / '_temp_data.csv'
        df.to_csv(temp_csv, index=False)
        
        analyzer = PitchAnalyzer(str(temp_csv))
        
        # 이상치 제거 (옵션)
        if remove_outliers:
            analyzer.remove_outliers(
                method=outlier_method,
                iqr_multiplier=iqr_multiplier,
                gender_specific=False  # 이미 성별로 분리된 경우
            )
        
        # 기술통계량 출력
        stats_df = analyzer.print_descriptive_statistics(
            group_col=group_col,
            save_path=str(output_path / 'descriptive_statistics.csv')
        )
        
        # 논문용 표 생성
        generate_paper_tables(analyzer, output_path, group_col, subset_name)
        
        # 감정별 구간 분석
        results = analyzer.analyze_emotion_by_bin(
            group_col=group_col,
            alpha=alpha,
            use_nonparametric=use_nonparametric
        )
        
        # 새 피쳐 분석 (기울기, 속도 등)
        if analyze_all_features:
            feature_results = analyzer.analyze_features_by_group(
                group_col=group_col,
                alpha=alpha,
                use_nonparametric=use_nonparametric
            )
            
            # 피쳐 분석 요약 저장
            feature_summary = analyzer.generate_feature_summary_table()
            if not feature_summary.empty:
                feature_summary.to_csv(output_path / 'feature_analysis_summary.csv', index=False)
        
        # 결과 출력
        summary = analyzer.generate_summary_table()
        print("\n" + "=" * 80)
        print(f"📊 [{subset_name}] 구간별 피치 통계 분석 결과 요약")
        print("=" * 80)
        print(summary.to_string(index=False))
        print("=" * 80)
        
        # 결과 저장
        analyzer.export_results(str(output_path))
        analyzer.plot_emotion_pitch_profile(str(output_path), subset_name=subset_name)
        
        # 모든 피쳐 시각화
        if analyze_all_features:
            analyzer.plot_all_features(str(output_path), group_col=group_col, subset_name=subset_name)
        
        # 통계 요약 저장
        summary.to_csv(output_path / 'statistical_summary.csv', index=False)
        
        # 임시 파일 삭제
        if temp_csv.exists():
            temp_csv.unlink()
        
        print(f"\n✅ [{subset_name}] 분석 완료!")
        
    finally:
        # stdout 복원 및 로그 파일 닫기
        sys.stdout = original_stdout
        log_file.close()
        logger.info(f"분석 로그 저장: {output_path / 'analysis_log.txt'}")
    
    return analyzer


def generate_paper_tables(analyzer, output_path: Path, group_col: str, subset_name: str):
    """
    논문용 기술통계량 표 생성
    
    생성되는 파일:
    - paper_table_descriptive.csv: 기술통계량 요약표
    - paper_table_descriptive.md: Markdown 형식
    - paper_table_descriptive.tex: LaTeX 형식
    - paper_table_statistical_tests.csv: 통계 검정 결과표
    """
    df = analyzer.df_filtered if analyzer.df_filtered is not None else analyzer.filter_valid_data()
    emotions = df[group_col].unique()
    
    print("\n" + "=" * 80)
    print(f"📄 [{subset_name}] 논문용 표 생성")
    print("=" * 80)
    
    # =========================================================================
    # 1. 데이터셋 기본 정보 표
    # =========================================================================
    dataset_info = []
    total_n = len(df)
    
    for emotion in emotions:
        emotion_df = df[df[group_col] == emotion]
        n = len(emotion_df)
        dataset_info.append({
            'Emotion': emotion,
            'N': n,
            'Percentage (%)': round(100 * n / total_n, 2)
        })
    
    dataset_info.append({
        'Emotion': 'Total',
        'N': total_n,
        'Percentage (%)': 100.0
    })
    
    dataset_df = pd.DataFrame(dataset_info)
    dataset_df.to_csv(output_path / 'paper_table_dataset_info.csv', index=False)
    
    print(f"\n▶ 데이터셋 정보:")
    print(dataset_df.to_string(index=False))
    
    # =========================================================================
    # 2. 주요 피쳐 기술통계량 표 (Mean ± SD 형식)
    # =========================================================================
    key_features = {
        'pitch_mean': 'Mean Pitch (Hz)',
        'pitch_std': 'Pitch SD (Hz)',
        'pitch_min': 'Min Pitch (Hz)',
        'pitch_max': 'Max Pitch (Hz)',
        'pitch_range': 'Pitch Range (Hz)',
        'pitch_slope': 'Overall Slope',
        'pitch_slope_first_half': 'First-half Slope',
        'pitch_slope_second_half': 'Second-half Slope',
        'pitch_velocity_abs_mean': 'Mean |Velocity| (Hz/s)',
        'pitch_inflection_count': 'Inflection Count',
        'pitch_peak_position': 'Peak Position (%)',
        'pitch_valley_position': 'Valley Position (%)',
        'pitch_point_count': 'Pitch Point Count'
    }
    
    # 존재하는 피쳐만 선택
    available_features = {k: v for k, v in key_features.items() if k in df.columns}
    
    descriptive_rows = []
    
    for feat_col, feat_name in available_features.items():
        row = {'Feature': feat_name}
        
        for emotion in emotions:
            emotion_df = df[df[group_col] == emotion]
            mean_val = emotion_df[feat_col].mean()
            std_val = emotion_df[feat_col].std()
            row[emotion] = f"{mean_val:.2f} ± {std_val:.2f}"
        
        # 전체 통계
        overall_mean = df[feat_col].mean()
        overall_std = df[feat_col].std()
        row['Total'] = f"{overall_mean:.2f} ± {overall_std:.2f}"
        
        descriptive_rows.append(row)
    
    descriptive_df = pd.DataFrame(descriptive_rows)
    
    # CSV 저장
    descriptive_df.to_csv(output_path / 'paper_table_descriptive.csv', index=False)
    
    # Markdown 저장
    md_content = f"# Descriptive Statistics - {subset_name}\n\n"
    md_content += "| Feature |"
    for emotion in emotions:
        md_content += f" {emotion} |"
    md_content += " Total |\n"
    md_content += "|" + "|".join(["---"] * (len(emotions) + 2)) + "|\n"
    
    for _, row in descriptive_df.iterrows():
        md_content += f"| {row['Feature']} |"
        for emotion in emotions:
            md_content += f" {row[emotion]} |"
        md_content += f" {row['Total']} |\n"
    
    with open(output_path / 'paper_table_descriptive.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # LaTeX 저장
    latex_content = f"% Descriptive Statistics - {subset_name}\n"
    latex_content += "\\begin{table}[htbp]\n"
    latex_content += "\\centering\n"
    latex_content += f"\\caption{{Descriptive Statistics of Pitch Features ({subset_name})}}\n"
    latex_content += "\\label{tab:descriptive_" + subset_name.replace(" ", "_").replace("(", "").replace(")", "") + "}\n"
    latex_content += "\\begin{tabular}{l" + "c" * (len(emotions) + 1) + "}\n"
    latex_content += "\\hline\n"
    latex_content += "Feature &"
    latex_content += " & ".join([str(e) for e in emotions])
    latex_content += " & Total \\\\\n"
    latex_content += "\\hline\n"
    
    for _, row in descriptive_df.iterrows():
        latex_content += row['Feature'].replace('%', '\\%') + " & "
        latex_content += " & ".join([row[e].replace('±', '$\\pm$') for e in emotions])
        latex_content += " & " + row['Total'].replace('±', '$\\pm$') + " \\\\\n"
    
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\end{table}\n"
    
    with open(output_path / 'paper_table_descriptive.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"\n▶ 기술통계량 표 (Mean ± SD):")
    print(descriptive_df.to_string(index=False))
    
    # =========================================================================
    # 3. 구간별 피치 평균 표
    # =========================================================================
    bin_rows = []
    bin_columns = [c for c in analyzer.BIN_COLUMNS if c in df.columns]
    
    for bin_col in bin_columns:
        # 구간명 정리 (예: pitch_bin_0_10 -> 0-10%)
        bin_name = bin_col.replace('pitch_bin_', '').replace('_', '-') + '%'
        row = {'Bin': bin_name}
        
        for emotion in emotions:
            emotion_df = df[df[group_col] == emotion]
            mean_val = emotion_df[bin_col].mean()
            std_val = emotion_df[bin_col].std()
            row[emotion] = f"{mean_val:.2f} ± {std_val:.2f}"
        
        bin_rows.append(row)
    
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(output_path / 'paper_table_bin_statistics.csv', index=False)
    
    # LaTeX 저장
    latex_bin = f"% Bin Statistics - {subset_name}\n"
    latex_bin += "\\begin{table}[htbp]\n"
    latex_bin += "\\centering\n"
    latex_bin += f"\\caption{{Mean Pitch by Utterance Position ({subset_name})}}\n"
    latex_bin += "\\label{tab:bin_" + subset_name.replace(" ", "_").replace("(", "").replace(")", "") + "}\n"
    latex_bin += "\\begin{tabular}{l" + "c" * len(emotions) + "}\n"
    latex_bin += "\\hline\n"
    latex_bin += "Position & " + " & ".join([str(e) for e in emotions]) + " \\\\\n"
    latex_bin += "\\hline\n"
    
    for _, row in bin_df.iterrows():
        latex_bin += row['Bin'] + " & "
        latex_bin += " & ".join([row[e].replace('±', '$\\pm$') for e in emotions]) + " \\\\\n"
    
    latex_bin += "\\hline\n"
    latex_bin += "\\end{tabular}\n"
    latex_bin += "\\end{table}\n"
    
    with open(output_path / 'paper_table_bin_statistics.tex', 'w', encoding='utf-8') as f:
        f.write(latex_bin)
    
    print(f"\n▶ 구간별 피치 평균 표:")
    print(bin_df.to_string(index=False))
    
    # =========================================================================
    # 4. 숫자만 있는 표 (Mean, SD 분리)
    # =========================================================================
    numeric_rows = []
    
    for feat_col, feat_name in available_features.items():
        row = {'Feature': feat_name}
        
        for emotion in emotions:
            emotion_df = df[df[group_col] == emotion]
            row[f'{emotion}_Mean'] = round(emotion_df[feat_col].mean(), 4)
            row[f'{emotion}_SD'] = round(emotion_df[feat_col].std(), 4)
            row[f'{emotion}_N'] = len(emotion_df)
        
        numeric_rows.append(row)
    
    numeric_df = pd.DataFrame(numeric_rows)
    numeric_df.to_csv(output_path / 'paper_table_numeric.csv', index=False)
    
    print(f"\n📁 논문용 표 저장 완료:")
    print(f"   - paper_table_dataset_info.csv: 데이터셋 정보")
    print(f"   - paper_table_descriptive.csv/md/tex: 기술통계량 (Mean ± SD)")
    print(f"   - paper_table_bin_statistics.csv/tex: 구간별 피치 평균")
    print(f"   - paper_table_numeric.csv: 숫자 분리형 표")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='감정별 피치 구간 통계 분석'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='analysis_output/pitch_analysis_data.csv',
        help='입력 CSV 파일 경로'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output/statistics',
        help='분석 결과 출력 디렉토리'
    )
    parser.add_argument(
        '--gender-separate',
        action='store_true',
        default=True,
        help='성별별 분리 분석 수행 (기본값: True)'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        default='emotion',
        help='그룹 변수 (emotion, style, speaker_gender 등)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='유의수준'
    )
    parser.add_argument(
        '--nonparametric',
        action='store_true',
        default=True,
        help='비모수적 검정 사용 (기본값: True)'
    )
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        default=True,
        help='이상치 제거 (기본값: True)'
    )
    parser.add_argument(
        '--outlier-method',
        type=str,
        default='iqr',
        choices=['iqr', 'zscore'],
        help='이상치 제거 방법 (iqr 또는 zscore)'
    )
    parser.add_argument(
        '--iqr-multiplier',
        type=float,
        default=1.5,
        help='IQR 배수 (기본값: 1.5)'
    )
    parser.add_argument(
        '--analyze-all-features',
        action='store_true',
        default=True,
        help='모든 피쳐에 대해 분석 (기울기, 속도 등)'
    )
    parser.add_argument(
        '--no-gender-separate',
        action='store_true',
        default=False,
        help='성별 분리 분석 비활성화'
    )
    
    args = parser.parse_args()
    
    # 원본 데이터 로드
    print("\n" + "=" * 80)
    print("🔬 피치 통계 분석 시작")
    print("=" * 80)
    
    original_df = pd.read_csv(args.input)
    print(f"📂 데이터 로드 완료: {len(original_df)}개 레코드")
    
    # 성별 컬럼 확인
    has_gender = 'speaker_gender' in original_df.columns
    if has_gender:
        gender_counts = original_df['speaker_gender'].value_counts()
        print(f"👫 성별 분포:")
        for gender, count in gender_counts.items():
            print(f"   - {gender}: {count}개 ({100*count/len(original_df):.1f}%)")
    
    # =========================================================================
    # 1. 전체 데이터 분석 (All)
    # =========================================================================
    all_output_dir = str(Path(args.output_dir) / 'All')
    run_analysis_for_subset(
        df=original_df,
        output_dir=all_output_dir,
        group_col=args.group_by,
        alpha=args.alpha,
        use_nonparametric=args.nonparametric,
        remove_outliers=args.remove_outliers,
        outlier_method=args.outlier_method,
        iqr_multiplier=args.iqr_multiplier,
        analyze_all_features=args.analyze_all_features,
        subset_name="전체 (All)"
    )
    
    # =========================================================================
    # 2. 성별별 분리 분석 (M / F)
    # =========================================================================
    if has_gender and args.gender_separate and not args.no_gender_separate:
        print("\n" + "=" * 80)
        print("👫 성별별 분리 분석 시작")
        print("   ℹ️ 남녀의 기본 음역대 차이(75Hz~600Hz)를 배제하여")
        print("      순수한 감정에 의한 피치 변화를 분석합니다.")
        print("=" * 80)
        
        # 성별 값 확인 (MALE/FEMALE 또는 M/F 형식 모두 지원)
        gender_values = original_df['speaker_gender'].unique()
        print(f"   발견된 성별 값: {gender_values}")
        
        # 남성 필터링 (MALE 또는 M)
        male_filter = original_df['speaker_gender'].isin(['MALE', 'M', 'male', 'm'])
        male_df = original_df[male_filter].copy()
        if len(male_df) > 0:
            male_output_dir = str(Path(args.output_dir) / 'M')
            run_analysis_for_subset(
                df=male_df,
                output_dir=male_output_dir,
                group_col=args.group_by,
                alpha=args.alpha,
                use_nonparametric=args.nonparametric,
                remove_outliers=args.remove_outliers,
                outlier_method=args.outlier_method,
                iqr_multiplier=args.iqr_multiplier,
                analyze_all_features=args.analyze_all_features,
                subset_name="남성 (M)"
            )
        else:
            print("⚠️ 남성 데이터가 없습니다.")
        
        # 여성 필터링 (FEMALE 또는 F)
        female_filter = original_df['speaker_gender'].isin(['FEMALE', 'F', 'female', 'f'])
        female_df = original_df[female_filter].copy()
        if len(female_df) > 0:
            female_output_dir = str(Path(args.output_dir) / 'F')
            run_analysis_for_subset(
                df=female_df,
                output_dir=female_output_dir,
                group_col=args.group_by,
                alpha=args.alpha,
                use_nonparametric=args.nonparametric,
                remove_outliers=args.remove_outliers,
                outlier_method=args.outlier_method,
                iqr_multiplier=args.iqr_multiplier,
                analyze_all_features=args.analyze_all_features,
                subset_name="여성 (F)"
            )
        else:
            print("⚠️ 여성 데이터가 없습니다.")
        
        # =====================================================================
        # 3. 남녀 비교 시각화 (Gender_Comparison 폴더)
        # =====================================================================
        if len(male_df) > 0 and len(female_df) > 0:
            # 필터링된 데이터로 비교 시각화 생성
            # 먼저 각 성별 데이터를 필터링 (유효 데이터만)
            BIN_COLUMNS = [f'pitch_bin_{i*10}_{(i+1)*10}' for i in range(10)]
            
            # 남성 데이터 필터링
            male_filtered = male_df.copy()
            male_filtered = male_filtered[male_filtered['pitch_point_count'] >= 3]
            for col in BIN_COLUMNS:
                if col in male_filtered.columns:
                    male_filtered = male_filtered[male_filtered[col] > 0]
            
            # 여성 데이터 필터링
            female_filtered = female_df.copy()
            female_filtered = female_filtered[female_filtered['pitch_point_count'] >= 3]
            for col in BIN_COLUMNS:
                if col in female_filtered.columns:
                    female_filtered = female_filtered[female_filtered[col] > 0]
            
            # 이상치 제거 (pitch_mean 기준 IQR)
            if args.remove_outliers:
                for df_to_filter, name in [(male_filtered, '남성'), (female_filtered, '여성')]:
                    if 'pitch_mean' in df_to_filter.columns:
                        Q1 = df_to_filter['pitch_mean'].quantile(0.25)
                        Q3 = df_to_filter['pitch_mean'].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - args.iqr_multiplier * IQR
                        upper = Q3 + args.iqr_multiplier * IQR
                        if name == '남성':
                            male_filtered = df_to_filter[(df_to_filter['pitch_mean'] >= lower) & 
                                                         (df_to_filter['pitch_mean'] <= upper)]
                        else:
                            female_filtered = df_to_filter[(df_to_filter['pitch_mean'] >= lower) & 
                                                           (df_to_filter['pitch_mean'] <= upper)]
            
            print(f"\n📊 남녀 비교 시각화용 데이터:")
            print(f"   - 남성: {len(male_filtered)}개")
            print(f"   - 여성: {len(female_filtered)}개")
            
            comparison_output_dir = str(Path(args.output_dir) / 'Gender_Comparison')
            generate_gender_comparison_plots(
                male_df=male_filtered,
                female_df=female_filtered,
                output_dir=comparison_output_dir,
                group_col=args.group_by
            )
    
    # =========================================================================
    # 4. 최종 요약 출력
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ 모든 분석 완료!")
    print("=" * 80)
    print(f"📁 결과 저장 위치:")
    print(f"   - 전체: {Path(args.output_dir) / 'All'}")
    if has_gender and args.gender_separate and not args.no_gender_separate:
        print(f"   - 남성: {Path(args.output_dir) / 'M'}")
        print(f"   - 여성: {Path(args.output_dir) / 'F'}")
        print(f"   - 남녀 비교: {Path(args.output_dir) / 'Gender_Comparison'}")
    print("=" * 80)
    print("\n📌 성별 분리 분석의 중요성:")
    print("   1. 남녀 기본 음역대 차이를 배제하여 순수 감정 효과 측정")
    print("   2. Hz 단위 동적 특성(속도/가속도)의 왜곡 방지")
    print("   3. 넓은 분포(variance)의 원인 규명 가능")
    print("   4. 성별 특화 감정 표현 패턴 발견 가능")
    
    # =========================================================================
    # 5. 논문용 주요 그림을 별도 폴더에 모아서 저장
    # =========================================================================
    import shutil
    paper_fig_dir = Path(args.output_dir) / 'paper_figures'
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 논문에 실릴 핵심 그림 목록 (소스 폴더, 파일명, 논문용 파일명)
    paper_figures = [
        # 전체 분석
        ('All', 'emotion_pitch_profile.png', 'fig_all_pitch_profile.png'),
        ('All', 'emotion_boxplot_by_bin.png', 'fig_all_boxplot_by_bin.png'),
        ('All', 'significance_barplot.png', 'fig_all_significance.png'),
        ('All', 'slope_features_comparison.png', 'fig_all_slope_features.png'),
        ('All', 'dynamics_features_comparison.png', 'fig_all_dynamics_features.png'),
        ('All', 'position_features_comparison.png', 'fig_all_position_features.png'),
        ('All', 'basic_stats_comparison.png', 'fig_all_basic_stats.png'),
        ('All', 'feature_heatmap.png', 'fig_all_feature_heatmap.png'),
        ('All', 'feature_significance.png', 'fig_all_feature_significance.png'),
        ('All', 'emotion_radar_chart.png', 'fig_all_radar_chart.png'),
        ('All', 'correlation_matrix.png', 'fig_all_correlation_matrix.png'),
        ('All', 'correlation_pearson.png', 'fig_all_correlation_pearson.png'),
        ('All', 'correlation_spearman.png', 'fig_all_correlation_spearman.png'),
        ('All', 'correlation_by_emotion.png', 'fig_all_correlation_by_emotion.png'),
        # 남성
        ('M', 'emotion_pitch_profile.png', 'fig_male_pitch_profile.png'),
        ('M', 'feature_heatmap.png', 'fig_male_feature_heatmap.png'),
        ('M', 'emotion_radar_chart.png', 'fig_male_radar_chart.png'),
        # 여성  
        ('F', 'emotion_pitch_profile.png', 'fig_female_pitch_profile.png'),
        ('F', 'feature_heatmap.png', 'fig_female_feature_heatmap.png'),
        ('F', 'emotion_radar_chart.png', 'fig_female_radar_chart.png'),
        # 남녀 비교
        ('Gender_Comparison', 'gender_comparison_pitch_profile.png', 'fig_gender_pitch_profile.png'),
        ('Gender_Comparison', 'gender_comparison_boxplot_by_bin.png', 'fig_gender_boxplot_by_bin.png'),
        ('Gender_Comparison', 'gender_comparison_pitch_mean.png', 'fig_gender_pitch_mean.png'),
        ('Gender_Comparison', 'gender_comparison_slope_features.png', 'fig_gender_slope_features.png'),
        ('Gender_Comparison', 'gender_comparison_radar_chart.png', 'fig_gender_radar_chart.png'),
        ('Gender_Comparison', 'gender_comparison_position_features.png', 'fig_gender_position_features.png'),
        ('Gender_Comparison', 'gender_comparison_dynamics_features.png', 'fig_gender_dynamics_features.png'),
    ]
    
    copied_count = 0
    for subfolder, src_name, dst_name in paper_figures:
        src = Path(args.output_dir) / subfolder / src_name
        if src.exists():
            shutil.copy2(str(src), str(paper_fig_dir / dst_name))
            copied_count += 1
    
    print(f"\n📄 논문용 그림 {copied_count}개를 별도 폴더에 저장:")
    print(f"   📁 {paper_fig_dir}")


if __name__ == '__main__':
    main()
