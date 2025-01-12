def decompose_hangul(char):
    """ 한글 음절을 초성, 중성, 종성으로 분리하는 함수 """
    hangul_start = 0xAC00
    hangul_end = 0xD7A3
    base_code, chosung, jungsung = 44032, 588, 28

    if hangul_start <= ord(char) <= hangul_end:
        temp = ord(char) - base_code
        chosung_index = temp // chosung
        jungsung_index = (temp - (chosung * chosung_index)) // jungsung
        jongsung_index = (temp - (chosung * chosung_index) - (jungsung * jungsung_index))
        return (chosung_index, jungsung_index, jongsung_index)
    else:
        return (None, None, None)

def is_hangul(char):
    """Check if a character is Hangul."""
    return '가' <= char <= '힣'

# 초성 리스트 (유니코드 초성순)
CHOSUNG_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 중성 리스트 (유니코드 중성순)
JUNGSEONG_LIST = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]