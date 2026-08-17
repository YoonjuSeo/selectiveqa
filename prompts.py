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


def build_target(gold_answer: str, evidence_span: str = None) -> str:
    """학습용 정답 출력(JSON 문자열)을 만든다. 파일럿은 응답가능 질문만 학습."""
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