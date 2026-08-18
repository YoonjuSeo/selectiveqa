# -*- coding: utf-8 -*-
"""
polarity_v2.py — yes_no 극성 매핑의 결함 수정 모듈 (후속 실험 단계 0)

결함 (본실험 보고서 7.3절 부수 발견):
    v1은 yes 집합을 먼저 검사하는 부분 문자열 매칭이어서
    '불가능' ⊃ '가능' 으로 긍정 오판된다. 재현 결과 이 결함은
    '아니오, 불가능합니다' 처럼 명시적 부정 표지가 공존해도
    긍정(1)을 반환한다 — yes 검사가 no 검사보다 먼저이기 때문.

수정 (v2): 최장 일치 우선(longest-match-first)
    yes/no 키워드를 합쳐 길이 내림차순으로 정렬한 뒤, 텍스트에
    등장하는 첫 번째(가장 긴) 키워드의 극성을 채택한다.
    '불가능'(3자)이 '가능'(2자)보다 먼저 검사되므로 오판이 사라지고,
    '예외 없이 불가능' 처럼 '예'가 다른 단어의 일부로 등장하는
    사례도 더 긴 '불가능'이 이겨 올바르게 판정된다.

동결 원칙:
    키워드 집합 자체는 v1과 동일하게 유지한다(사전등록 채점 정의의
    최소 수정). 키워드 추가·삭제는 별도 민감도로만 다룬다.

사용:
    from polarity_v2 import polarity_v1, polarity_v2, is_correct_v2
    # evaluate.py 패치 시에는 is_correct 의 yes_no 분기를
    # polarity_v2 호출로 교체 (아래 PATCH_NOTE 참조).
"""
import string
import unicodedata

# ---- evaluate.py 와 동일한 정규화 (동결) ----------------------------------
_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}


def normalize(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if not ch.isspace() and ch not in _PUNCT)


# ---- 키워드 집합 (v1과 동일, 동결) ----------------------------------------
YES_WORDS = {"예", "네", "맞습니다", "yes", "true", "가능"}
NO_WORDS = {"아니오", "아니요", "아닙니다", "no", "false", "불가능"}

# 정규화된 (키워드, 극성) 목록 — 길이 내림차순, 동률 시 사전순(결정성 보장)
_KEYWORDS_V2 = sorted(
    [(normalize(w), 1) for w in YES_WORDS] + [(normalize(w), 0) for w in NO_WORDS],
    key=lambda kv: (-len(kv[0]), kv[0]),
)


def polarity_v1(text):
    """v1 로직의 충실한 재현 (diff 산출 전용). 반환: 1 | 0 | None"""
    t = normalize(text)
    if any(normalize(w) in t for w in YES_WORDS):
        return 1
    if any(normalize(w) in t for w in NO_WORDS):
        return 0
    return None


def polarity_v2(text, return_match=False):
    """최장 일치 우선 극성 판정.

    반환: polarity (1|0|None)
          return_match=True 이면 (polarity, matched_keyword, ambiguous)
          ambiguous: 채택 키워드와 다른 극성의 키워드가 텍스트에
                     공존하는지 여부 (판정에는 불사용, 감사 로그용)
    """
    t = normalize(text)
    matched = None
    for kw, pol in _KEYWORDS_V2:
        if kw in t:
            matched = (kw, pol)
            break
    if matched is None:
        return (None, None, False) if return_match else None

    kw, pol = matched
    if not return_match:
        return pol

    # 감사용: 반대 극성 키워드의 공존 검사 (채택 키워드 구간을 제거한 뒤 검사
    # — '불가능' 채택 시 그 내부의 '가능'을 반대 극성 공존으로 세지 않기 위함)
    residual = t.replace(kw, "\u0000")
    ambiguous = any(
        k in residual for k, p in _KEYWORDS_V2 if p != pol
    )
    return pol, kw, ambiguous


def is_correct_v2(pred, gold, qtype, tol=None, is_correct_fallback=None):
    """yes_no 만 v2 극성으로 채점하고, 그 외 유형·미표지 폴백은
    기존 is_correct 로 위임한다.

    is_correct_fallback: evaluate.is_correct (다른 유형과 문자열 폴백의
                         동결 로직 재사용). 미지정 시 정규화 일치만 사용.
    """
    if qtype == "yes_no":
        gp, pp = polarity_v2(gold), polarity_v2(pred)
        if gp is not None and pp is not None:
            return gp == pp
        # 미표지 폴백: v1과 동일하게 정규화 문자열 일치
        return normalize(pred) == normalize(gold)
    if is_correct_fallback is not None:
        return is_correct_fallback(pred, gold, qtype, tol)
    return normalize(pred) == normalize(gold)


PATCH_NOTE = """
evaluate.py 최소 패치 (is_correct 내부 yes_no 분기 교체):

    if qtype == "yes_no":
-       yes = {"예", "네", "맞습니다", "yes", "true", "가능"}
-       no = {"아니오", "아니요", "아닙니다", "no", "false", "불가능"}
-       def polarity(t): ...  (부분 문자열, yes 우선)
+       from polarity_v2 import polarity_v2 as polarity  # 최장 일치 우선
        gp, pp = polarity(gold), polarity(pred)
        if gp is not None and pp is not None:
            return gp == pp
        return normalize(pred) == normalize(gold)

패치는 rescore_yesno_diff.py 로 판정 불변을 확인한 뒤 적용한다.
"""


if __name__ == "__main__":
    cases = [
        "불가능합니다", "불가능", "아니오, 불가능합니다", "네, 가능합니다",
        "아닙니다", "예외 없이 불가능", "가능", "예", "판단할 수 없습니다",
    ]
    print(f"{'입력':<22} v1  v2  (일치키워드, 공존)")
    for s in cases:
        v2, kw, amb = polarity_v2(s, return_match=True)
        print(f"{s:<22} {str(polarity_v1(s)):<3} {str(v2):<3} ({kw}, amb={amb})")