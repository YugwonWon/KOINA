#!/usr/bin/env python3
"""수정 전후 phoneme_kr 매핑 비교 스크립트"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lib"))

from textgrid import TextGrid
from utils.jamo import decompose_hangul, is_hangul, CHOSUNG_LIST

OLD_DIR = Path(__file__).resolve().parent / "test_results"
NEW_DIR = Path(__file__).resolve().parent / "test_results_fixed"

def get_onset_mapping(tg_path):
    """TextGrid에서 syllable+phoneme_kr 정보를 추출하여 초성 매핑 반환"""
    tg = TextGrid.fromFile(str(tg_path))
    syl_tier = pkr_tier = utt_tier = None
    for tier in tg.tiers:
        if tier.name == "syllable":
            syl_tier = tier
        elif tier.name == "phoneme_kr":
            pkr_tier = tier
        elif tier.name == "utterance":
            utt_tier = tier
    
    if not syl_tier or not pkr_tier:
        return [], ""
    
    transcript = ""
    if utt_tier:
        transcript = " ".join(iv.mark for iv in utt_tier.intervals if iv.mark.strip())
    
    results = []
    for siv in syl_tier.intervals:
        if not siv.mark.strip():
            continue
        ch = siv.mark.strip()
        if not is_hangul(ch):
            continue
        cho_i, _, _ = decompose_hangul(ch)
        if cho_i is None:
            continue
        expected = CHOSUNG_LIST[cho_i]
        
        # 이 음절 구간에 속하는 phoneme_kr 중 첫 번째 비어있지 않은 것
        first_kr = None
        for piv in pkr_tier.intervals:
            if piv.minTime >= siv.minTime - 0.003 and piv.maxTime <= siv.maxTime + 0.003:
                if piv.mark.strip():
                    first_kr = piv.mark.strip()
                    break
        
        if first_kr and first_kr != expected:
            # 초성 ㅇ은 모음으로 시작하므로 매핑 불일치가 정상일 수 있음
            results.append((ch, expected, first_kr))
    
    return results, transcript

def main():
    old_folders = sorted([d for d in OLD_DIR.iterdir() if d.is_dir()])
    
    total_fixed = 0
    total_new_issues = 0
    total_old_issues = 0
    total_new_remaining = 0
    
    print("=" * 100)
    print(f"{'파일':<30} {'수정 전 오류':>10} {'수정 후 오류':>10} {'개선':>8} {'악화':>8}")
    print("=" * 100)
    
    details_fixed = []
    details_worse = []
    
    for old_folder in old_folders:
        name = old_folder.name
        new_folder = NEW_DIR / name
        
        old_tg = list(old_folder.glob("*.TextGrid"))
        new_tg = list(new_folder.glob("*.TextGrid")) if new_folder.exists() else []
        
        if not old_tg or not new_tg:
            continue
        
        # _pct.TextGrid 제외
        old_tg = [t for t in old_tg if "_pct" not in t.name]
        new_tg = [t for t in new_tg if "_pct" not in t.name]
        
        if not old_tg or not new_tg:
            continue
        
        old_issues, transcript = get_onset_mapping(old_tg[0])
        new_issues, _ = get_onset_mapping(new_tg[0])
        
        old_set = set(old_issues)
        new_set = set(new_issues)
        
        fixed = old_set - new_set
        worse = new_set - old_set
        
        total_old_issues += len(old_issues)
        total_new_remaining += len(new_issues)
        total_fixed += len(fixed)
        total_new_issues += len(worse)
        
        status = ""
        if fixed:
            status = "✓ 개선"
        if worse:
            status += " ✗ 악화"
        
        if fixed or worse:
            print(f"{name:<30} {len(old_issues):>10} {len(new_issues):>10} {len(fixed):>8} {len(worse):>8}  {status}")
            
            for ch, expected, actual in fixed:
                details_fixed.append((name, ch, expected, actual))
            for ch, expected, actual in worse:
                details_worse.append((name, ch, expected, actual))
    
    print("=" * 100)
    print(f"{'합계':<30} {total_old_issues:>10} {total_new_remaining:>10} {total_fixed:>8} {total_new_issues:>8}")
    print()
    
    if details_fixed:
        print(f"\n✓ 수정으로 개선된 매핑 ({len(details_fixed)}건):")
        print(f"  {'파일':<30} {'음절':>4} {'기대':>4} {'수정전':>6} → {'수정후':>6}")
        for name, ch, expected, old_actual in details_fixed:
            print(f"  {name:<30} {ch:>4} {expected:>4} {old_actual:>6} → {expected:>6}")
    
    if details_worse:
        print(f"\n✗ 수정으로 악화된 매핑 ({len(details_worse)}건):")
        print(f"  {'파일':<30} {'음절':>4} {'기대':>4} {'수정전':>6} → {'수정후':>6}")
        for name, ch, expected, new_actual in details_worse:
            # 수정 전 값 찾기
            old_folder = OLD_DIR / name
            old_tg = [t for t in old_folder.glob("*.TextGrid") if "_pct" not in t.name]
            if old_tg:
                old_issues, _ = get_onset_mapping(old_tg[0])
                old_match = [a for c, e, a in old_issues if c == ch and e == expected]
                old_val = old_match[0] if old_match else "?"
            else:
                old_val = "?"
            print(f"  {name:<30} {ch:>4} {expected:>4} {old_val:>6} → {new_actual:>6}")
    
    print(f"\n총 초성 매핑 오류: {total_old_issues} → {total_new_remaining} ({total_old_issues - total_new_remaining}건 감소)")

if __name__ == "__main__":
    main()
