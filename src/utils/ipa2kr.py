import re
from textwrap import shorten

IPA2KR = {
    # ────────── 파열/파찰음 계열 ──────────
    "b":"ㅂ",    "bʲ":"ㅂ",   "bʷ":"ㅂ",
    "p":"ㅂ",    "pː":"ㅂ",   "p̚":"ㅂ",
    "pʰ":"ㅍ",   "pʰː":"ㅍ",
    "p͈":"ㅃ",   "p͈ʲ":"ㅃ",
    "pʲ":"ㅂ",   "pʲː":"ㅂ", "pʷ":"ㅂ",
    
    "c":"ㅈ",    "cː":"ㅈ",   "cʰ":"ㅊ", "cʰː":"ㅊ", "c͈":"ㅉ",
    "d":"ㄷ",    "dʲ":"ㄷ",   "dʷ":"ㄷ",
    "dʑ":"ㅈ",   "dʑʷ":"ㅈ",
    "ɟ" : "ㅈ",   # 유성 경구개 파열음
    
    "k":"ㄱ",    "kː":"ㄱ",   "k̚":"ㄱ",
    "kʰ":"ㅋ",   "kʰː":"ㅋ",
    "k͈":"ㄲ",   "k͈ː":"ㄲ",
    "kʷ":"ㅋ",   "kʷː":"ㅋ",
    "k͈ʷ":"ㄲ",
    "ɡ" : "ㄱ",   # 유성 연구개 파열음
    "ɡʷ": "ㄱ",   # 원순화된 /ɡ/ → 실제 표면엔 ‘구’ 계열로 실현돼도 자모는 ㄱ
    
    "s":"ㅅ",    "sː":"ㅅ",
    "sʰ":"ㅅ",   "sʰː":"ㅅ",
    "s͈":"ㅆ",   "s͈ʷ":"ㅆ",
    "sʷ":"ㅅ",
    "ɕ": "ㅅ",    # 무성 경구개 마찰음 (plain)
    "ɕʰ": "ㅅ",   # 유기 경구개 마찰음 → '시' 가까운 음
    "ɕ͈": "ㅆ",   # 된소리 경구개 마찰음
    
    "t":"ㄷ",    "tː":"ㄷ",   "t̚":"ㅆ",
    "tʰ":"ㅌ",   "tʰː":"ㅌ",
    "t͈":"ㄸ",   "t͈ː":"ㄸ",  "t͈ʲ":"ㄸ",
    "tʲ":"ㄷ",   "tʷ":"ㄷ",   "tʷː":"ㄷ",
    
    # 치경-구개 파찰음
    "tɕ":"ㅈ",   "tɕː":"ㅈ",
    "tɕʰ":"ㅊ",  "tɕʰː":"ㅊ",
    "tɕ͈":"ㅉ",  "tɕ͈ː":"ㅉ",
    "tɕʷ":"ㅈ",  "tɕʷː":"ㅈ",
    "tɕ͈ʷ":"ㅉ",

    # ────────── 비음/류음/마찰음 ──────────
    "m":"ㅁ",    "mː":"ㅁ",   "mʲ":"ㅁ",  "mʲː":"ㅁ",
    "n":"ㄴ",    "nː":"ㄴ",
    "ŋ":"ㅇ",
    "ɲ":"ㄴ",    # 연구개 /ɲ/ → ‘니’ (모음과 결합하여 이중모음)
    "ɾ":"ㄹ",    "ɾʲ":"ㄹ",  "ɾʷ":"ㄹ",
    "ɭ":"ㄹ",    "ɭː":"ㄹ",
    "ʎ":"ㄹ",    "ʎː":"ㄹ",
    "h":"ㅎ",    "ɦ":"ㅎ",   "x":"ㅎ",  "ç":"ㅎ",
    "ɣ":"ㅎ",    "β":"ㅂ",   "βʷ":"ㅂ",
    "ɸ":"ㅍ",    "ɸʷ":"ㅍ",

    # ────────── 반모음/기타 ──────────
    "j":"ㅣ",    "w":"ㅜ",    "ɥ":"ㅟ",  "ɰ":"ㅡ",

    # ────────── 모음(길이 ‧ 원순성 단순화) ──────────
    "i":"ㅣ",    "iː":"ㅣ",
    "e":"ㅔ",    "eː":"ㅔ",
    "ɛ":"ㅐ",    "ɛː":"ㅐ",
    "ɨ":"ㅡ",    "ɨː":"ㅡ",
    "ɐ":"ㅏ",
    "ʌ":"ㅓ",    "ʌː":"ㅓ",
    "o":"ㅗ",    "oː":"ㅗ",
    "u":"ㅜ",    "uː":"ㅜ",
    "ʝ" : "ㅣ",   # 유성 경구개 마찰음 → 모음 /i/에 가까운 glide 계열
}

DICT_PHONES = [
                'b',
                'bʲ',
                'bʷ',
                'c',
                'cʰ',
                'cʰː',
                'c͈',
                'd',
                'dʑ',
                'dʑʷ',
                'dʲ',
                'dʷ',
                'e',
                'eː',
                'h',
                'i',
                'iː',
                'j',
                'k',
                'kʰ',
                'kʰː',
                'kʷ',
                'kʷː',
                'k̚',
                'k͈',
                'k͈ʷ',
                'k͈ː',
                'm',
                'mʲ',
                'mʲː',
                'mː',
                'n',
                'nː',
                'o',
                'oː',
                'p',
                'pʰ',
                'pʰː',
                'pʲ',
                'pʲː',
                'pʷ',
                'p̚',
                'p͈',
                'p͈ʲ',
                's',
                'sʰ',
                'sʰː',
                'sʷ',
                'sː',
                's͈',
                's͈ʷ',
                't',
                'tɕ',
                'tɕʰ',
                'tɕʰː',
                'tɕʷ',
                'tɕʷː',
                'tɕː',
                'tɕ͈',
                'tɕ͈ʷ',
                'tɕ͈ː',
                'tʰ',
                'tʰː',
                'tʲ',
                'tʷ',
                'tʷː',
                't̚',
                't͈',
                't͈ʲ',
                't͈ː',
                'u',
                'uː',
                'w',
                'x',
                'ç',
                'ŋ',
                'ɐ',
                'ɕ',
                'ɕʰ',
                'ɕ͈',
                'ɛ',
                'ɛː',
                'ɟ',
                'ɡ',
                'ɡʷ',
                'ɣ',
                'ɥ',
                'ɦ',
                'ɨ',
                'ɨː',
                'ɭ',
                'ɭː',
                'ɰ',
                'ɲ',
                'ɸ',
                'ɸʷ',
                'ɾ',
                'ɾʲ',
                'ɾʷ',
                'ʌ',
                'ʌː',
                'ʎ',
                'ʎː',
                'ʝ',
                'β',
                'βʷ'
            ]

# ────────────────────────────────────────────────────────────────────
# 반모음 + 모음 → 이중모음 병합 테이블
# ────────────────────────────────────────────────────────────────────

# j-계 이중모음 (경구개 접근음 j 또는 경구개 비음 ɲ 뒤의 모음)
J_VOWEL_MERGE = {
    'ʌ': 'ㅕ',  'ʌː': 'ㅕ',
    'o': 'ㅛ',  'oː': 'ㅛ',
    'u': 'ㅠ',  'uː': 'ㅠ',
    'ɐ': 'ㅑ',
    'e': 'ㅖ',  'eː': 'ㅖ',
    'ɛ': 'ㅒ',  'ɛː': 'ㅒ',
    'i': 'ㅣ',  'iː': 'ㅣ',   # j+i = ㅣ (동일하지만 구간 병합)
}

# w-계 이중모음 (양순 접근음 w 뒤의 모음)
W_VOWEL_MERGE = {
    'ɐ': 'ㅘ',
    'ʌ': 'ㅝ',  'ʌː': 'ㅝ',
    'e': 'ㅞ',  'eː': 'ㅞ',
    'ɛ': 'ㅙ',  'ɛː': 'ㅙ',
    'i': 'ㅟ',  'iː': 'ㅟ',
}

# ɰ-계 이중모음 (연구개 접근음 ɰ 뒤의 모음)
VELAR_APPROX_MERGE = {
    'i': 'ㅢ',  'iː': 'ㅢ',
}


def ipa2kr(p):
    """
    Convert IPA symbol to Korean Hangul character.
    """
    return IPA2KR.get(p, p)


def ipa_sequence_to_kr(phones):
    """
    IPA 음소 시퀀스를 한글 자모 시퀀스로 변환합니다.
    반모음(j, w, ɰ)과 후행 모음을 이중모음으로 병합하고,
    경구개 비음(ɲ) 뒤의 모음을 j-이중모음으로 변환합니다.

    Args:
        phones: list of dict — {'start', 'end', 'text'} (IPA phone 구간)

    Returns:
        list of dict — {'start', 'end', 'text'} (한글 자모 구간, 이중모음 병합 적용)
    """
    result = []
    i = 0
    while i < len(phones):
        current = phones[i]
        text = current.get('text', '') or ''

        # 빈 텍스트(묵음 구간)는 그대로 통과
        if text == '':
            result.append({'start': current['start'],
                           'end':   current['end'],
                           'text':  ''})
            i += 1
            continue

        next_phone = phones[i + 1] if i + 1 < len(phones) else None
        next_text  = (next_phone.get('text', '') or '') if next_phone else ''

        # ── 1. ɲ (경구개 비음) + 모음 → ㄴ + j-이중모음 ──
        #    ɲ 자체가 /nj/ 를 인코딩하므로, 후행 모음에 j-glide 반영
        if text == 'ɲ' and next_text in J_VOWEL_MERGE:
            result.append({'start': current['start'],
                           'end':   current['end'],
                           'text':  'ㄴ'})
            result.append({'start': next_phone['start'],
                           'end':   next_phone['end'],
                           'text':  J_VOWEL_MERGE[next_text]})
            i += 2
            continue

        # ── 2. j (경구개 접근음) + 모음 → 이중모음 (구간 병합) ──
        if text == 'j' and next_text in J_VOWEL_MERGE:
            result.append({'start': current['start'],
                           'end':   next_phone['end'],
                           'text':  J_VOWEL_MERGE[next_text]})
            i += 2
            continue

        # ── 3. w (양순 접근음) + 모음 → 이중모음 (구간 병합) ──
        if text == 'w' and next_text in W_VOWEL_MERGE:
            result.append({'start': current['start'],
                           'end':   next_phone['end'],
                           'text':  W_VOWEL_MERGE[next_text]})
            i += 2
            continue

        # ── 4. ɰ (연구개 접근음) + i → ㅢ (구간 병합) ──
        if text == 'ɰ' and next_text in VELAR_APPROX_MERGE:
            result.append({'start': current['start'],
                           'end':   next_phone['end'],
                           'text':  VELAR_APPROX_MERGE[next_text]})
            i += 2
            continue

        # ── 5. 기본: 1:1 매핑 ──
        result.append({'start': current['start'],
                       'end':   current['end'],
                       'text':  IPA2KR.get(text, text)})
        i += 1

    return result


# ────────────────────────────────────────────────────────────────────
# 형태 정보(ground truth) 기반 IPA→한글 보정 + 음절 경계 결정
# ────────────────────────────────────────────────────────────────────

# 이중모음의 선행 glide IPA
_DIPHTHONG_GLIDE = {
    'ㅑ': 'j', 'ㅒ': 'j', 'ㅕ': 'j', 'ㅖ': 'j', 'ㅛ': 'j', 'ㅠ': 'j',
    'ㅘ': 'w', 'ㅙ': 'w', 'ㅚ': 'w', 'ㅝ': 'w', 'ㅞ': 'w', 'ㅟ': 'w',
    'ㅢ': 'ɰ',
}

# IPA glide 음소 집합
_GLIDE_IPA = {'j', 'w', 'ɰ', 'ɥ'}

# 겹받침(복합 종성) 분해 테이블
_DOUBLE_JONGSEONG = {
    'ㄳ': ('ㄱ', 'ㅅ'), 'ㄵ': ('ㄴ', 'ㅈ'), 'ㄶ': ('ㄴ', 'ㅎ'),
    'ㄺ': ('ㄹ', 'ㄱ'), 'ㄻ': ('ㄹ', 'ㅁ'), 'ㄼ': ('ㄹ', 'ㅂ'),
    'ㄽ': ('ㄹ', 'ㅅ'), 'ㄾ': ('ㄹ', 'ㅌ'), 'ㄿ': ('ㄹ', 'ㅍ'),
    'ㅀ': ('ㄹ', 'ㅎ'), 'ㅄ': ('ㅂ', 'ㅅ'),
}

# 묵음/무음으로 취급할 IPA mark
_SILENT_MARKS = {'', 'sil', 'sp'}

_EPS = 0.005  # 5 ms boundary tolerance

# IPA 모음 기저형 (길이 표시 ː 제거 후 판별용)
_VOWEL_IPA_BASE = {'i', 'e', 'ɛ', 'ɨ', 'ɐ', 'ʌ', 'o', 'u', 'ʝ'}


def _is_vowel_ipa(text):
    """IPA 음소가 모음인지 판별 (길이 표시(ː) 무시)"""
    return text.rstrip('ː') in _VOWEL_IPA_BASE


def build_kr_and_syllables(phones, words_restored):
    """
    형태 정보(word text)를 ground truth로 삼아
    1. IPA 음소 → 한글 자모 변환 보정 (phoneme_kr)
    2. 음절 경계 결정 (syllable intervals)

    어절 텍스트의 자모 분해 결과를 기준으로 MFA IPA 음소를 greedy 정렬하여,
    ɲ→ㅇ(종성), d→ㅈ 등의 음운 변동을 형태에 맞게 보정합니다.

    Args:
        phones: list[dict]  — {'start', 'end', 'text'} MFA 음소 (빈 text = 묵음)
        words_restored: list[dict] — {'start', 'end', 'text'} 어절 복원 결과

    Returns:
        (phones_kr, syllables)
        phones_kr:  list[dict] — 보정된 한글 자모 구간
        syllables:  list[dict] — 음절 구간 {'start', 'end', 'text'}
    """
    from utils.jamo import (decompose_hangul, is_hangul,
                            CHOSUNG_LIST, JUNGSEONG_LIST, JONGSEONG_LIST)

    # phone_idx → 보정된 한글 라벨 (None = 이중모음 병합 대상)
    assignment = {}
    syllables = []

    for word in words_restored:
        w_text = word.get('text', '')
        if not w_text:
            continue
        w_start, w_end = word['start'], word['end']

        # ── 이 어절에 속하는 실음소 인덱스 수집 ──
        word_pis = []
        for pi, p in enumerate(phones):
            if pi in assignment:
                continue
            pt = (p.get('text', '') or '').strip()
            if pt.lower() in _SILENT_MARKS or pt == 'spn':
                continue
            if p['start'] >= w_start - _EPS and p['end'] <= w_end + _EPS:
                word_pis.append(pi)

        # ── 어절 텍스트 → 음절·자모 분해 ──
        syl_specs = []  # list of (char, [(position, jamo)])
        for ch in w_text:
            if is_hangul(ch):
                cho_i, jung_i, jong_i = decompose_hangul(ch)
                if cho_i is None:
                    syl_specs.append((ch, [('other', ch)]))
                    continue
                cho  = CHOSUNG_LIST[cho_i]
                jung = JUNGSEONG_LIST[jung_i]
                jong = JONGSEONG_LIST[jong_i] if jong_i > 0 else None

                parts = [('onset', cho), ('nucleus', jung)]
                if jong:
                    if jong in _DOUBLE_JONGSEONG:
                        j1, j2 = _DOUBLE_JONGSEONG[jong]
                        parts.append(('coda', j1))
                        parts.append(('coda2', j2))
                    else:
                        parts.append(('coda', jong))
                syl_specs.append((ch, parts))
            else:
                syl_specs.append((ch, [('other', ch)]))

        # ── greedy alignment ──
        cursor = 0          # word_pis 내 커서
        j_consumed = False   # ɲ(=/nj/) 초성이 j를 이미 포함했는지
        ok = True
        syl_result = []      # [(syl_char, [(phone_idx, kr_label)])]

        for syl_char, parts in syl_specs:
            syl_phones = []  # (phone_idx, label)

            for pos, jamo in parts:
                # ── 초성 ㅇ (무음) → phone 소비 없음 ──
                if pos == 'onset' and jamo == 'ㅇ':
                    j_consumed = False
                    continue

                # ── 초성 자음 ──
                if pos == 'onset':
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    # 호환성: 초성에는 자음 IPA만 할당 가능
                    if _is_vowel_ipa(pt):
                        ok = False; break
                    # ɲ는 /nj/를 인코딩 → 뒤의 j-이중모음에서 glide 소비 불필요
                    j_consumed = (pt == 'ɲ')
                    syl_phones.append((pi, jamo))
                    cursor += 1

                # ── 중성(모음) ──
                elif pos == 'nucleus':
                    glide = _DIPHTHONG_GLIDE.get(jamo)

                    if glide and not j_consumed:
                        # 이중모음: glide+vowel(2) 또는 단독 vowel(1)
                        if cursor >= len(word_pis):
                            ok = False; break
                        pi = word_pis[cursor]
                        pt = phones[pi]['text']
                        # 호환성: 중성에는 모음 또는 glide만 할당 가능
                        if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                            ok = False; break

                        if pt in _GLIDE_IPA and cursor + 1 < len(word_pis):
                            # glide + vowel → 하나의 이중모음으로 병합
                            pi2 = word_pis[cursor + 1]
                            syl_phones.append((pi, jamo))    # glide → 이중모음 라벨
                            syl_phones.append((pi2, None))   # vowel → 병합 마커
                            cursor += 2
                        else:
                            # glide 없이 단독 vowel
                            syl_phones.append((pi, jamo))
                            cursor += 1

                    elif glide and j_consumed:
                        # ɲ가 /nj/ → j 이미 소비됨, vowel만 취함
                        if cursor >= len(word_pis):
                            ok = False; break
                        pi = word_pis[cursor]
                        pt = phones[pi]['text']
                        # 호환성: 모음 IPA만 허용
                        if not _is_vowel_ipa(pt):
                            ok = False; break
                        syl_phones.append((pi, jamo))
                        cursor += 1
                        j_consumed = False

                    else:
                        # 단모음: 1 phone
                        if cursor >= len(word_pis):
                            ok = False; break
                        pi = word_pis[cursor]
                        pt = phones[pi]['text']
                        # 호환성: 중성에는 모음 또는 glide만 할당 가능
                        if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                            ok = False; break
                        syl_phones.append((pi, jamo))
                        cursor += 1
                        j_consumed = False

                # ── 종성(받침) ──
                elif pos in ('coda', 'coda2'):
                    if cursor < len(word_pis):
                        pi = word_pis[cursor]
                        pt = phones[pi]['text']
                        # 호환성: 종성에는 자음 IPA만 할당 가능
                        if _is_vowel_ipa(pt):
                            # 종성에 모음이 오면 → 해당 종성은 연음 등으로 사라진 것
                            # phone을 소비하지 않고 건너뜀
                            pass
                        else:
                            syl_phones.append((pi, jamo))
                            cursor += 1
                    else:
                        # 종성 phone 부족 → 연음/축약으로 사라진 경우
                        # ok 유지, 해당 종성은 건너뜀
                        pass

                # ── 기타 (비한글 문자) ──
                elif pos == 'other':
                    if cursor < len(word_pis):
                        pi = word_pis[cursor]
                        syl_phones.append((pi, jamo))
                        cursor += 1

            if not ok:
                break
            syl_result.append((syl_char, syl_phones))

        # 남은 phone이 있으면 alignment 실패
        if cursor < len(word_pis):
            ok = False

        # ── 정렬 결과 반영 ──
        if ok and syl_result:
            for syl_char, sp in syl_result:
                for pi, label in sp:
                    assignment[pi] = label
                # 음절 구간
                if sp:
                    pis = [pi for pi, _ in sp]
                    syllables.append({
                        'start': phones[pis[0]]['start'],
                        'end':   phones[pis[-1]]['end'],
                        'text':  syl_char
                    })
        else:
            # fallback: 기존 IPA2KR + count 기반 음절 분할
            for pi in word_pis:
                pt = phones[pi]['text']
                assignment[pi] = IPA2KR.get(pt, pt)
            _fallback_syllables(syl_specs, word_pis, phones, syllables)

    # ── phones_kr 구성 ──
    phones_kr = []
    for i, p in enumerate(phones):
        pt = (p.get('text', '') or '').strip()

        if pt.lower() in _SILENT_MARKS:
            phones_kr.append({'start': p['start'], 'end': p['end'], 'text': ''})
        elif pt == 'spn':
            phones_kr.append({'start': p['start'], 'end': p['end'], 'text': 'spn'})
        elif i in assignment:
            label = assignment[i]
            if label is None:
                # 이중모음 병합: 직전 항목의 end 확장
                if phones_kr and phones_kr[-1]['text']:
                    phones_kr[-1]['end'] = p['end']
            else:
                phones_kr.append({'start': p['start'], 'end': p['end'], 'text': label})
        else:
            # 어디에도 할당되지 않은 phone → 기존 매핑
            phones_kr.append({
                'start': p['start'], 'end': p['end'],
                'text': IPA2KR.get(pt, pt)
            })

    return phones_kr, syllables


def _fallback_syllables(syl_specs, word_pis, phones, syllables):
    """
    greedy 정렬 실패 시 fallback: 자모 count 기반 음절 분할.
    초성 ㅇ 제외 자모 개수만큼 phone을 배분합니다.
    """
    cursor = 0
    for ch, parts in syl_specs:
        n = sum(1 for pos, j in parts if not (pos == 'onset' and j == 'ㅇ'))
        if n <= 0:
            n = 1
        if cursor + n > len(word_pis):
            n = len(word_pis) - cursor
        if n <= 0:
            continue
        first_pi = word_pis[cursor]
        last_pi  = word_pis[cursor + n - 1]
        syllables.append({
            'start': phones[first_pi]['start'],
            'end':   phones[last_pi]['end'],
            'text':  ch
        })
        cursor += n