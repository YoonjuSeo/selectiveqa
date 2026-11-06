# -*- coding: utf-8 -*-
"""
prompts.py — 프롬프트 템플릿 (학습과 추론이 반드시 이 모듈을 공유할 것)

M0(zero-shot)과 M1(파인튜닝) 조건이 동일한 프롬프트를 사용해야
프롬프트 차이가 보정(calibration) 차이에 섞이지 않는다.
"""

import json

SYSTEM_PROMPT = (
    "당신은 금융 문서 질의응답 시스템입니다. "
    "주어진 지문에 근거해서만 답하세요. "
    "반드시 아래 JSON 형식 한 줄로만 출력하세요.\n"
    '{"answerable": true, "answer": "<답>", "evidence_span": "<지문 내 근거 문장>"}\n'
    "지문으로 답할 수 없는 질문이면:\n"
    '{"answerable": false, "answer": null, "evidence_span": null}'
)


def build_messages(context: str, question: str) -> list:
    """tokenizer.apply_chat_template 에 넣을 messages 리스트를 만든다."""
    user = f"[지문]\n{context}\n\n[질문]\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


UA_TARGETS = {
    # 기존(사전등록) 무응답 타깃 — 시스템 프롬프트에 명시된 형식 그대로
    "null":   {"answerable": False, "answer": None, "evidence_span": None},
    # 1단계 ablation(보고서 4.6절): 확답과 같은 슬롯 구조를 유지하고 값만 짧은 문자열로 채움
    "filled": {"answerable": False, "answer": "답 없음",
               "evidence_span": "지문에 해당 정보 없음"},
}


def build_target(gold_answer: str, evidence_span: str = None,
                 answerable: bool = True, ua_style: str = "null") -> str:
    """학습용 정답 출력(JSON 문자열)을 만든다. answerable=False 면
    ua_style 에 해당하는 무응답 타깃을 사용한다 (기본 "null" = 기존과 동일)."""
    if not answerable:
        return json.dumps(UA_TARGETS[ua_style], ensure_ascii=False)
    obj = {
        "answerable": True,
        "answer": gold_answer,
        "evidence_span": evidence_span if evidence_span else gold_answer,
    }
    return json.dumps(obj, ensure_ascii=False)


import re

_ANSWERABLE_RE = re.compile(r'"answerable"\s*:\s*(true|false)')
_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*('
    r'"(?:[^"\\]|\\.)*"'      # 문자열 (이스케이프 허용)
    r'|null'
    r'|-?\d+(?:\.\d+)?'        # 숫자
    r')')


def parse_model_output(text: str) -> dict:
    """모델 출력에서 JSON을 최대한 관대하게 파싱한다.

    1차: 첫 '{' ~ 마지막 '}' 구간 json.loads.
    2차: max_new_tokens 절단 등으로 JSON이 닫히지 않은 경우,
         answerable/answer 필드를 정규식으로 구조 복구.
         (출력 필드 순서가 answerable → answer → evidence_span 이므로
          절단은 대개 evidence_span에서 발생하고 answer는 온전함.)
    반환: {"answerable": bool|None, "answer": str|None, "parse_ok": bool}
    """
    text = text.strip()
    # 코드블록/앞뒤 잡음 제거 후 첫 '{' ~ 마지막 '}' 구간 시도
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            return {
                "answerable": obj.get("answerable"),
                "answer": obj.get("answer"),
                "parse_ok": True,
            }
        except json.JSONDecodeError:
            pass
    # 2차: 절단된 JSON에서 필드 단위 복구
    m_ans = _ANSWERABLE_RE.search(text)
    m_val = _ANSWER_RE.search(text)
    if m_ans and m_val:
        raw = m_val.group(1)
        if raw == "null":
            answer = None
        else:
            try:
                answer = json.loads(raw)      # 문자열 이스케이프/숫자 해석
            except json.JSONDecodeError:
                answer = raw.strip('"')
        return {
            "answerable": m_ans.group(1) == "true",
            "answer": answer,
            "parse_ok": True,                 # 모델은 형식을 지킴 (절단은 디코딩 한도 탓)
        }
    # 파싱 실패 시 원문 전체를 답으로 간주 (채점 시 대부분 오답 처리됨)
    return {"answerable": None, "answer": text, "parse_ok": False}

def apply_template(tokenizer, messages, enable_thinking=None):
    """학습·추론 공용 템플릿 적용. Qwen3: enable_thinking=False 로 non-thinking 고정.
    None 이면 kwarg 자체를 넘기지 않아 EXAONE 프롬프트가 기존과 바이트 단위로 동일."""
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kw)


# ---------------------------------------------------------------------------
# closed-book 통제 조건 (2026-11-06, docs/prereg_closedbook_261106.md)
# 지문 없이 질문만 주어 모델의 사전지식만으로 답하게 한다. 기존 SYSTEM_PROMPT·build_messages는
# 바이트 단위로 그대로 두고(학습·본 추론 공유), 아래 함수는 --closed-book 옵션에서만 쓰인다.
#   force   : 모르더라도 가장 가능성 높은 답을 반드시 쓰게 함 → 사전지식 정답률의 상한(주 조건)
#   abstain : 모르면 무응답 형식을 허용 → 탐색적(모델이 '모른다'고 아는지)
# 출력 JSON 형식은 기존과 같게 유지해 parse_model_output·answer 구간 신뢰도 계산을 그대로 재사용한다.
# ---------------------------------------------------------------------------
CLOSED_BOOK_SYSTEM_PROMPT = {
    "force": (
        "당신은 금융 질의응답 시스템입니다. "
        "이번에는 지문이 주어지지 않습니다. 당신이 이미 알고 있는 지식만으로 질문에 답하세요. "
        "확실하지 않더라도 가장 가능성이 높은 답을 하나 반드시 쓰세요. "
        "반드시 아래 JSON 형식 한 줄로만 출력하세요.\n"
        '{"answerable": true, "answer": "<답>", "evidence_span": null}'
    ),
    "abstain": (
        "당신은 금융 질의응답 시스템입니다. "
        "이번에는 지문이 주어지지 않습니다. 당신이 이미 알고 있는 지식만으로 질문에 답하세요. "
        "반드시 아래 JSON 형식 한 줄로만 출력하세요.\n"
        '{"answerable": true, "answer": "<답>", "evidence_span": null}\n'
        "답을 모르면:\n"
        '{"answerable": false, "answer": null, "evidence_span": null}'
    ),
}


def build_messages_closed_book(question: str, mode: str = "force") -> list:
    """closed-book 조건의 messages. 지문([지문] 블록)을 넣지 않고 질문만 준다.
    user 턴의 [질문] 표기는 build_messages와 같게 유지한다."""
    if mode not in CLOSED_BOOK_SYSTEM_PROMPT:
        raise ValueError(f"closed-book mode는 {list(CLOSED_BOOK_SYSTEM_PROMPT)} 중 하나: {mode}")
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT[mode]},
        {"role": "user", "content": f"[질문]\n{question}"},
    ]
