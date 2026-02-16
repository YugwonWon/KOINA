#!/usr/bin/env python3
"""KOINA 결과 TextGrid 파일들의 정렬(align) 및 음절 정렬 품질을 테스트"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lib'))

from textgrid import TextGrid

BASE = "/data3/yugwon/auto-trans-k-intonation"
RESULTS_DIR = os.path.join(BASE, "tests/test_results")

results = sorted(os.listdir(RESULTS_DIR))
print(f"{'='*80}")
print(f"KOINA 테스트 결과 분석 ({len(results)}개 파일)")
print(f"{'='*80}\n")

total_files = 0
pass_files = 0
issues = []

for folder in results:
    folder_path = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # TextGrid 파일 찾기 (pct 아닌 기본 파일)
    tg_files = [f for f in os.listdir(folder_path) if f.endswith('.TextGrid') and '_pct' not in f]
    if not tg_files:
        issues.append((folder, "TextGrid 파일 없음"))
        continue
    
    tg_path = os.path.join(folder_path, tg_files[0])
    total_files += 1
    file_issues = []
    
    try:
        tg = TextGrid.fromFile(tg_path)
        tier_names = [t.name for t in tg.tiers]
        
        print(f"[{total_files:2d}] {folder}")
        print(f"     티어: {tier_names}")
        
        # 각 티어 분석
        tier_stats = {}
        for tier in tg.tiers:
            if hasattr(tier, 'intervals'):  # IntervalTier
                intervals = [iv for iv in tier.intervals if iv.mark and iv.mark.strip() and iv.mark.strip() != 'SP']
                total_intervals = len(tier.intervals)
                active_intervals = len(intervals)
                tier_stats[tier.name] = {
                    'type': 'Interval',
                    'total': total_intervals,
                    'active': active_intervals,
                }
                
                # 정렬 일관성 확인: 시간 순서 확인
                prev_end = 0
                time_order_ok = True
                overlaps = 0
                gaps = 0
                for iv in tier.intervals:
                    if iv.minTime < prev_end - 0.001:
                        overlaps += 1
                        time_order_ok = False
                    elif iv.minTime > prev_end + 0.001:
                        gaps += 1
                    prev_end = iv.maxTime
                
                if overlaps > 0:
                    file_issues.append(f"  {tier.name}: {overlaps}개 겹침(overlap)")
                
            elif hasattr(tier, 'points'):  # PointTier
                tier_stats[tier.name] = {
                    'type': 'Point',
                    'total': len(tier.points),
                }
        
        # 결과 출력
        for name, stats in tier_stats.items():
            if stats['type'] == 'Interval':
                print(f"     {name:15s}: {stats['active']:4d}/{stats['total']:4d} 활성 구간")
            else:
                print(f"     {name:15s}: {stats['total']:4d} 포인트")
        
        # word 티어 샘플 출력
        for tier in tg.tiers:
            if tier.name == 'word' and hasattr(tier, 'intervals'):
                words = [(iv.minTime, iv.maxTime, iv.mark) for iv in tier.intervals if iv.mark and iv.mark.strip() and iv.mark != 'SP']
                if words:
                    print(f"     어절 샘플: {words[:3]}")
                else:
                    file_issues.append("  word 티어에 어절 없음")
                break
        
        # syllable 티어 확인
        for tier in tg.tiers:
            if tier.name == 'syllable' and hasattr(tier, 'intervals'):
                syllables = [(iv.minTime, iv.maxTime, iv.mark) for iv in tier.intervals if iv.mark and iv.mark.strip() and iv.mark != 'SP']
                if syllables:
                    print(f"     음절 샘플: {syllables[:5]}")
                else:
                    file_issues.append("  syllable 티어에 음절 없음")
                break
        
        # phoneme_kr 티어 확인
        for tier in tg.tiers:
            if tier.name == 'phoneme_kr' and hasattr(tier, 'intervals'):
                phonemes_kr = [(iv.minTime, iv.maxTime, iv.mark) for iv in tier.intervals if iv.mark and iv.mark.strip() and iv.mark != 'SP']
                if phonemes_kr:
                    print(f"     한글음소 샘플: {phonemes_kr[:5]}")
                else:
                    file_issues.append("  phoneme_kr 티어에 한글음소 없음")
                break
        
        # Points 티어 확인 (Momel)
        for tier in tg.tiers:
            if tier.name == 'Points' and hasattr(tier, 'points'):
                if len(tier.points) == 0:
                    file_issues.append("  Points 티어 비어있음 (Momel 실패?)")
                break
        
        if file_issues:
            print(f"     ⚠ 이슈: {'; '.join(file_issues)}")
            issues.append((folder, file_issues))
        else:
            pass_files += 1
            print(f"     ✓ 정상")
        
        print()
        
    except Exception as e:
        file_issues.append(f"TextGrid 파싱 오류: {e}")
        issues.append((folder, file_issues))
        print(f"[{total_files:2d}] {folder} - 오류: {e}\n")

# 요약
print(f"\n{'='*80}")
print(f"요약")
print(f"{'='*80}")
print(f"총 파일: {total_files}")
print(f"정상: {pass_files}")
print(f"이슈: {len(issues)}")
print(f"성공률: {100*pass_files/total_files:.1f}%")

if issues:
    print(f"\n이슈 목록:")
    for folder, issue_list in issues:
        if isinstance(issue_list, list):
            for iss in issue_list:
                print(f"  {folder}: {iss}")
        else:
            print(f"  {folder}: {issue_list}")
