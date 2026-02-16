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


# ────────────────────────────────────────────────────────────────────
# G2P(Grapheme-to-Phoneme) 기반 정렬 지원
# ────────────────────────────────────────────────────────────────────

_g2p_instance = None
_g2p_available = None


def _get_g2p():
    """G2P 인스턴스를 lazy-load합니다. (g2pk2 패키지 필요)"""
    global _g2p_instance, _g2p_available
    if _g2p_available is None:
        try:
            from g2pk2 import G2p
            _g2p_instance = G2p()
            _g2p_available = True
        except ImportError:
            _g2p_available = False
    return _g2p_instance


# 종성 호환성 테이블: 음운변동(비음화, 중화 등)으로 인한 변이 허용
_CODA_COMPAT = {
    'ㄱ': {'ㄱ', 'ㅋ', 'ㄲ'},
    'ㄷ': {'ㄷ', 'ㅌ', 'ㅅ', 'ㅆ', 'ㅈ', 'ㅊ', 'ㄸ'},
    'ㅂ': {'ㅂ', 'ㅍ', 'ㅃ'},
    'ㅁ': {'ㅁ'},
    'ㄴ': {'ㄴ', 'ㄹ'},
    'ㄹ': {'ㄹ', 'ㄴ'},
    'ㅇ': {'ㅇ', 'ㄴ', 'ㅁ'},  # 비음 위치동화
    'ㅎ': {'ㅎ'},
    'ㅅ': {'ㅅ', 'ㅆ', 'ㄷ', 'ㅌ'},
    'ㅆ': {'ㅆ', 'ㅅ', 'ㄷ', 'ㅌ'},
}


def _is_coda_compatible(ipa_text, expected_jamo):
    """IPA 음소가 기대 종성 자모와 호환되는지 확인합니다."""
    actual_kr = IPA2KR.get(ipa_text, '')
    if actual_kr == expected_jamo:
        return True
    compat_set = _CODA_COMPAT.get(expected_jamo, {expected_jamo})
    return actual_kr in compat_set


def _g2p_align_word(w_text, word_pis, phones):
    """
    G2P(grapheme-to-phoneme)를 활용한 어절 단위 정렬.

    발음 변환 결과의 자모를 기준으로 MFA IPA 음소와 정렬합니다.
    연음, 비음화, 경음화, 겹받침 간소화 등 모든 음운 규칙을 자동 처리합니다.

    Args:
        w_text:    어절 텍스트 (원형)
        word_pis:  이 어절에 속하는 phones 인덱스 리스트
        phones:    전체 phone 리스트 [{'start', 'end', 'text'}, ...]

    Returns:
        (ok, assignments, syllables)
        ok:          정렬 성공 여부
        assignments: {phone_idx: kr_label} (None = 이중모음 병합 대상)
        syllables:   [{'start', 'end', 'text'}, ...]
    """
    g2p = _get_g2p()
    if g2p is None:
        return False, None, None

    from utils.jamo import (decompose_hangul, is_hangul,
                            CHOSUNG_LIST, JUNGSEONG_LIST, JONGSEONG_LIST)

    # G2P 발음 변환
    try:
        pron_text = g2p(w_text)
    except Exception:
        return False, None, None

    # 한글 문자만 추출
    orig_chars = [ch for ch in w_text if is_hangul(ch)]
    pron_chars = [ch for ch in pron_text if is_hangul(ch)]

    # 음절 수 불일치 → 실패 (fallback으로 전환)
    if len(orig_chars) != len(pron_chars) or not pron_chars:
        return False, None, None

    # 발음형 자모 분해
    pron_syl_specs = []
    for ch in pron_chars:
        cho_i, jung_i, jong_i = decompose_hangul(ch)
        if cho_i is None:
            pron_syl_specs.append((ch, [('other', ch)]))
            continue
        cho  = CHOSUNG_LIST[cho_i]
        jung = JUNGSEONG_LIST[jung_i]
        jong = JONGSEONG_LIST[jong_i] if jong_i > 0 else None

        parts = [('onset', cho), ('nucleus', jung)]
        if jong:
            parts.append(('coda', jong))
        pron_syl_specs.append((ch, parts))

    # ── Greedy alignment (발음 자모 기준) ──
    cursor = 0
    ok = True
    syl_result = []  # [(orig_char, [(phone_idx, kr_label)])]

    for syl_idx, (pron_char, parts) in enumerate(pron_syl_specs):
        orig_char = orig_chars[syl_idx]
        syl_phones = []

        for pos, jamo in parts:
            # 초성 ㅇ (무음) → phone 소비 없음
            if pos == 'onset' and jamo == 'ㅇ':
                continue

            # 초성 자음
            if pos == 'onset':
                if cursor >= len(word_pis):
                    ok = False; break
                pi = word_pis[cursor]
                pt = phones[pi]['text']
                if _is_vowel_ipa(pt):
                    ok = False; break
                syl_phones.append((pi, jamo))
                cursor += 1

            # 중성 (모음)
            elif pos == 'nucleus':
                glide = _DIPHTHONG_GLIDE.get(jamo)

                if glide:
                    # 이중모음
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                        ok = False; break

                    if pt in _GLIDE_IPA and cursor + 1 < len(word_pis):
                        # glide + vowel → 이중모음 병합
                        pi2 = word_pis[cursor + 1]
                        syl_phones.append((pi, jamo))
                        syl_phones.append((pi2, None))  # 병합 마커
                        cursor += 2
                    else:
                        # 단독 vowel → 이중모음 라벨
                        syl_phones.append((pi, jamo))
                        cursor += 1
                else:
                    # 단모음
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                        ok = False; break
                    syl_phones.append((pi, jamo))
                    cursor += 1

            # 종성 (받침) — 호환성 검사 후 선택적 소비
            elif pos == 'coda':
                if cursor < len(word_pis):
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if _is_vowel_ipa(pt):
                        pass  # 종성 탈락 (다음 phone이 모음)
                    elif _is_coda_compatible(pt, jamo):
                        syl_phones.append((pi, jamo))
                        cursor += 1
                    else:
                        pass  # 종성 탈락/변동으로 비호환 → 건너뜀
                # else: phone 부족 → 종성 생략

            # 기타 (비한글)
            elif pos == 'other':
                if cursor < len(word_pis):
                    pi = word_pis[cursor]
                    syl_phones.append((pi, jamo))
                    cursor += 1

        if not ok:
            break
        syl_result.append((orig_char, syl_phones))

    # 남은 phone이 있으면 정렬 실패
    if cursor < len(word_pis):
        ok = False

    if not syl_result:
        return False, None, None

    # 결과 구성
    assignments = {}
    syllables = []
    for orig_char, sp in syl_result:
        for pi, label in sp:
            assignments[pi] = label
        if sp:
            pis = [pi for pi, _ in sp]
            syllables.append({
                'start': phones[pis[0]]['start'],
                'end':   phones[pis[-1]]['end'],
                'text':  orig_char
            })

    # 부분 정렬: ok=False 이면 성공한 음절까지만 반환 (caller 가 나머지를 fallback)
    return ok, assignments, syllables


def _make_syl_specs(text):
    """텍스트(또는 문자 리스트)를 음절 스펙 [(char, [(pos, jamo), ...]), ...] 으로 분해합니다."""
    from utils.jamo import (decompose_hangul, is_hangul,
                            CHOSUNG_LIST, JUNGSEONG_LIST, JONGSEONG_LIST)
    specs = []
    for ch in text:
        if is_hangul(ch):
            cho_i, jung_i, jong_i = decompose_hangul(ch)
            if cho_i is None:
                specs.append((ch, [('other', ch)]))
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
            specs.append((ch, parts))
        else:
            specs.append((ch, [('other', ch)]))
    return specs


def _try_greedy(syl_specs, word_pis, phones):
    """
    음절 스펙에 대해 greedy alignment을 시도합니다.

    Returns:
        syl_result:  [(syl_char, [(phone_idx, kr_label)])]
        ok:          bool — 전체 성공 여부
    """
    cursor = 0
    j_consumed = False
    ok = True
    syl_result = []

    for syl_char, parts in syl_specs:
        syl_phones = []

        for pos, jamo in parts:
            if pos == 'onset' and jamo == 'ㅇ':
                j_consumed = False
                continue

            if pos == 'onset':
                if cursor >= len(word_pis):
                    ok = False; break
                pi = word_pis[cursor]
                pt = phones[pi]['text']
                if _is_vowel_ipa(pt):
                    ok = False; break
                j_consumed = (pt == 'ɲ')
                syl_phones.append((pi, jamo))
                cursor += 1

            elif pos == 'nucleus':
                glide = _DIPHTHONG_GLIDE.get(jamo)

                if glide and not j_consumed:
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                        ok = False; break
                    if pt in _GLIDE_IPA and cursor + 1 < len(word_pis):
                        pi2 = word_pis[cursor + 1]
                        syl_phones.append((pi, jamo))
                        syl_phones.append((pi2, None))
                        cursor += 2
                    else:
                        syl_phones.append((pi, jamo))
                        cursor += 1

                elif glide and j_consumed:
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if not _is_vowel_ipa(pt):
                        ok = False; break
                    syl_phones.append((pi, jamo))
                    cursor += 1
                    j_consumed = False

                else:
                    if cursor >= len(word_pis):
                        ok = False; break
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if not _is_vowel_ipa(pt) and pt not in _GLIDE_IPA:
                        ok = False; break
                    syl_phones.append((pi, jamo))
                    cursor += 1
                    j_consumed = False

            elif pos in ('coda', 'coda2'):
                if cursor < len(word_pis):
                    pi = word_pis[cursor]
                    pt = phones[pi]['text']
                    if _is_vowel_ipa(pt):
                        pass
                    else:
                        syl_phones.append((pi, jamo))
                        cursor += 1

            elif pos == 'other':
                if cursor < len(word_pis):
                    pi = word_pis[cursor]
                    syl_phones.append((pi, jamo))
                    cursor += 1

        if not ok:
            break
        syl_result.append((syl_char, syl_phones))

    if cursor < len(word_pis):
        ok = False

    return syl_result, ok


def _fill_spn_gaps(w_text, phones, assignment, syllables, w_start, w_end, spn_expansion):
    """
    어절 내 spn 구간에 해당하는 누락 음절/자모를 텍스트 기반으로 후보정합니다.

    greedy/G2P 정렬이 비-spn 음소에 대해 완료된 후 호출합니다.
    spn 구간에 속할 음절을 찾아 시간 비례 분할하여 syllables 와 spn_expansion 에 추가합니다.

    Args:
        w_text:         어절 텍스트 (한글)
        phones:         전체 phone 리스트
        assignment:     phone_idx → 한글 라벨 (in-place)
        syllables:      음절 리스트 (in-place)
        w_start, w_end: 어절 시간 범위
        spn_expansion:  {phone_idx: [(start, end, label), ...]}  (in-place)
    """
    from utils.jamo import (decompose_hangul, is_hangul,
                            CHOSUNG_LIST, JUNGSEONG_LIST, JONGSEONG_LIST)

    expected_chars = [ch for ch in w_text if is_hangul(ch)]
    if not expected_chars:
        return

    # ── 어절 내 spn 구간 수집 ──
    spn_phones = []
    for pi, p in enumerate(phones):
        pt = (p.get('text', '') or '').strip()
        if pt == 'spn' and p['start'] >= w_start - _EPS and p['end'] <= w_end + _EPS:
            spn_phones.append((pi, p))
    if not spn_phones:
        return

    # ── 이미 매칭된 음절 목록 (시간순) ──
    word_syls = sorted(
        [s for s in syllables
         if s['start'] >= w_start - _EPS and s['end'] <= w_end + _EPS],
        key=lambda s: s['start']
    )
    if len(word_syls) >= len(expected_chars):
        return                      # 음절 모두 존재

    # ── expected ↔ matched 순서보존 매핑 ──
    matched_positions = []          # expected 내 인덱스
    search_from = 0
    for ms in word_syls:
        for i in range(search_from, len(expected_chars)):
            if expected_chars[i] == ms['text']:
                matched_positions.append(i)
                search_from = i + 1
                break

    covered = set(matched_positions)
    missing_indices = [i for i in range(len(expected_chars)) if i not in covered]
    if not missing_indices:
        return

    # ── 연속 누락 그룹 ──
    groups = []
    cur_grp = [missing_indices[0]]
    for idx in missing_indices[1:]:
        if idx == cur_grp[-1] + 1:
            cur_grp.append(idx)
        else:
            groups.append(cur_grp)
            cur_grp = [idx]
    groups.append(cur_grp)

    # ── 각 누락 그룹을 spn 구간에 매핑 ──
    used_spn = set()
    for group in groups:
        missing_chars = [expected_chars[i] for i in group]

        # 그룹 앞뒤 시간 범위 결정
        prev_end   = w_start
        next_start = w_end
        for mp_idx, mp in enumerate(matched_positions):
            if mp < group[0]:
                prev_end = max(prev_end, word_syls[mp_idx]['end'])
        for mp_idx, mp in enumerate(matched_positions):
            if mp > group[-1]:
                next_start = min(next_start, word_syls[mp_idx]['start'])
                break

        # 시간 범위와 겹치는 spn 모두 수집 (연속 spn 병합)
        target_spns = []
        for pi, p in spn_phones:
            if pi in used_spn:
                continue
            if p['end'] > prev_end - _EPS and p['start'] < next_start + _EPS:
                target_spns.append((pi, p))
                used_spn.add(pi)
        if not target_spns:
            for pi, p in spn_phones:
                if pi not in used_spn:
                    target_spns.append((pi, p))
                    used_spn.add(pi)
                    break
        if not target_spns:
            continue

        # 병합된 spn 시간 범위
        first_spn_pi = target_spns[0][0]
        spn_start = min(p['start'] for _, p in target_spns)
        spn_end   = max(p['end']   for _, p in target_spns)
        n_chars   = len(missing_chars)
        dur_each  = (spn_end - spn_start) / n_chars

        # ── 시간 비례 분할 → syllables + phoneme_kr 자모 ──
        jamo_entries = []
        for k, ch in enumerate(missing_chars):
            sub_start = spn_start + k * dur_each
            # float 정밀도: 마지막 항목은 spn_end 를 정확히 사용
            sub_end   = spn_end if k == n_chars - 1 \
                        else spn_start + (k + 1) * dur_each

            syllables.append({'start': sub_start, 'end': sub_end, 'text': ch})

            cho_i, jung_i, jong_i = decompose_hangul(ch)
            if cho_i is not None:
                cho  = CHOSUNG_LIST[cho_i]
                jung = JUNGSEONG_LIST[jung_i]
                jong = JONGSEONG_LIST[jong_i] if jong_i > 0 else None

                jamo_parts = []
                if cho != 'ㅇ':
                    jamo_parts.append(cho)
                jamo_parts.append(jung)
                if jong:
                    jamo_parts.append(jong)

                n_parts  = max(1, len(jamo_parts))
                part_dur = (sub_end - sub_start) / n_parts
                t = sub_start
                for j, jamo in enumerate(jamo_parts):
                    t_end = sub_end if j == n_parts - 1 else t + part_dur
                    jamo_entries.append((t, t_end, jamo))
                    t = t_end
            else:
                jamo_entries.append((sub_start, sub_end, ch))

        if jamo_entries:
            # 첫 spn에 전체 자모 배정, 나머지 spn은 빈 리스트 (출력 생략 마커)
            spn_expansion.setdefault(first_spn_pi, []).extend(jamo_entries)
            for pi, _ in target_spns[1:]:
                spn_expansion.setdefault(pi, [])   # 빈 리스트 = 출력 생략


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
    spn_expansion = {}   # spn 후보정: {phone_idx: [(start, end, label), ...]}

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

        # ── G2P 기반 정렬 우선 시도 ──
        g2p_ok, g2p_assign, g2p_syls = _g2p_align_word(w_text, word_pis, phones)
        if g2p_ok:
            assignment.update(g2p_assign)
            syllables.extend(g2p_syls)
            _fill_spn_gaps(w_text, phones, assignment, syllables,
                           w_start, w_end, spn_expansion)
            continue

        # ── G2P 부분 정렬 결과가 있으면 보존 후 나머지만 fallback ──
        if g2p_assign is not None:
            assignment.update(g2p_assign)
            syllables.extend(g2p_syls)
            unassigned_pis = [pi for pi in word_pis if pi not in assignment]
            if unassigned_pis:
                for pi in unassigned_pis:
                    pt = phones[pi]['text']
                    assignment[pi] = IPA2KR.get(pt, pt)
                matched_count = len(g2p_syls)
                hangul_remaining = [ch for ch in w_text if is_hangul(ch)][matched_count:]
                remaining_specs = _make_syl_specs(hangul_remaining)
                _fallback_syllables(remaining_specs, unassigned_pis, phones, syllables)
            _fill_spn_gaps(w_text, phones, assignment, syllables,
                           w_start, w_end, spn_expansion)
            continue

        # ── (fallback) 어절 텍스트 → 음절·자모 분해 ──
        syl_specs = _make_syl_specs(w_text)

        # ── spn 존재 여부 확인 ──
        word_has_spn = False
        for pi, p in enumerate(phones):
            pt = (p.get('text', '') or '').strip()
            if pt == 'spn' and p['start'] >= w_start - _EPS and p['end'] <= w_end + _EPS:
                word_has_spn = True
                break

        # ── greedy alignment (spn이 있으면 세그먼트별 정렬) ──
        if word_has_spn and len(syl_specs) > 0:
            # ── 어절 내 phones를 spn 기준으로 세그먼트 분리 ──
            word_all_pis = []         # [(pi, is_spn)]
            for pi, p in enumerate(phones):
                if pi in assignment:
                    continue
                pt = (p.get('text', '') or '').strip()
                if pt.lower() in _SILENT_MARKS:
                    continue
                if p['start'] >= w_start - _EPS and p['end'] <= w_end + _EPS:
                    word_all_pis.append((pi, pt == 'spn'))

            # 연속된 real/spn 세그먼트로 그룹화
            segments = []  # [{'type': 'real'|'spn', 'pis': [pi, ...]}]
            for pi, is_spn in word_all_pis:
                seg_type = 'spn' if is_spn else 'real'
                if segments and segments[-1]['type'] == seg_type:
                    segments[-1]['pis'].append(pi)
                else:
                    segments.append({'type': seg_type, 'pis': [pi]})

            # 각 real 세그먼트를 적절한 음절 범위에 매칭
            # spn 세그먼트는 syl_cursor를 전진시키지 않음 —
            # sliding start가 IPA-한글 호환성으로 올바른 음절을 찾고,
            # _fill_spn_gaps가 누락 음절을 spn 시간대에 배치합니다.
            syl_cursor = 0      # syl_specs 내 현재 위치
            syl_result = []
            ok = True

            for seg in segments:
                if seg['type'] == 'spn':
                    continue    # syl_cursor 전진 없이 건너뜀

                # real 세그먼트: syl_cursor 부터 sliding start
                seg_pis = seg['pis']
                best_seg_result = []
                best_seg_start = syl_cursor
                best_seg_ok = False

                for start_idx in range(
                        max(0, syl_cursor), len(syl_specs)):
                    trial_result, trial_ok = _try_greedy(
                        syl_specs[start_idx:], seg_pis, phones)
                    if trial_ok:
                        best_seg_result = trial_result
                        best_seg_start = start_idx
                        best_seg_ok = True
                        break
                    if len(trial_result) > len(best_seg_result):
                        best_seg_result = trial_result
                        best_seg_start = start_idx

                syl_result.extend(best_seg_result)
                syl_cursor = best_seg_start + len(best_seg_result)
                if not best_seg_ok:
                    ok = False
        else:
            # spn이 없으면 기존대로 처음부터 greedy
            syl_result, ok = _try_greedy(syl_specs, word_pis, phones)

        # ── 정렬 결과 반영 (부분 정렬도 보존) ──
        if syl_result:
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
        if not ok:
            # 미할당 phone 에 대해서만 IPA2KR fallback
            unassigned_pis = [pi for pi in word_pis if pi not in assignment]
            if unassigned_pis:
                for pi in unassigned_pis:
                    pt = phones[pi]['text']
                    assignment[pi] = IPA2KR.get(pt, pt)
                remaining_specs = syl_specs[len(syl_result):]
                _fallback_syllables(remaining_specs, unassigned_pis, phones, syllables)

        # ── spn 후보정: 누락된 음절을 spn 구간에서 복원 ──
        _fill_spn_gaps(w_text, phones, assignment, syllables,
                       w_start, w_end, spn_expansion)

    # ── 음절을 시간순 정렬 (spn 후보정으로 추가된 항목 포함) ──
    syllables.sort(key=lambda s: s['start'])

    # ── phones_kr 구성 ──
    phones_kr = []
    for i, p in enumerate(phones):
        pt = (p.get('text', '') or '').strip()

        if pt.lower() in _SILENT_MARKS:
            phones_kr.append({'start': p['start'], 'end': p['end'], 'text': ''})
        elif pt == 'spn':
            if i in spn_expansion:
                for sub_start, sub_end, label in spn_expansion[i]:
                    phones_kr.append({'start': sub_start, 'end': sub_end, 'text': label})
            else:
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

    # ── float 정밀도 보정: 인접 구간 경계 스냅 ──
    for idx in range(1, len(phones_kr)):
        if abs(phones_kr[idx - 1]['end'] - phones_kr[idx]['start']) < 1e-6:
            phones_kr[idx]['start'] = phones_kr[idx - 1]['end']

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