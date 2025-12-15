#!/usr/bin/env python3
"""
v4 최종 버전 v2 - 보수적이지만 효과적인 접근
====================================================
전략 변경:
- 3문장 제한 제거 (v4에는 역효과)
- 중복 제거 + 간결화에 집중
- v4의 정보량을 최대한 유지하면서 품질 개선
"""

import pandas as pd
import re

print("\n" + "="*80)
print("🎯 v4 최종 버전 v2 - 보수적이지만 효과적인 접근")
print("="*80)

# 파일 로드
v4 = pd.read_csv('./prediction/submit_solar_v4.csv')
v3_micro = pd.read_csv('./prediction/submit_solar_v3_microtuned.csv')

print(f"\n📂 로드 완료:")
print(f"  v4: {len(v4)}개")
print(f"  v3_microtuned: {len(v3_micro)}개")

def focused_micro_tune_v4(summary: str) -> str:
    """
    집중 미세조정 전략
    
    핵심: v4의 정보량을 유지하면서 품질만 개선
    - 중복 제거
    - 불필요한 표현 정리
    - 문장 완성도 유지
    - 3문장 제한 없음 (v4는 정보량이 강점)
    """
    
    # 1) 기본 공백 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # 2) 명백한 중복 제거
    summary = re.sub(r'(\S+)\s+\1', r'\1', summary)  # 같은 단어 연속 제거
    
    # 조사 중복
    summary = re.sub(r'에게\s+에게', '에게', summary)
    summary = re.sub(r'에서\s+에서', '에서', summary)
    summary = re.sub(r'에\s+에\s', '에 ', summary)
    
    # 동사 중복
    summary = re.sub(r'합니다\s+합니다', '합니다', summary)
    summary = re.sub(r'한다\s+한다', '한다', summary)
    summary = re.sub(r'하고\s+하고', '하고', summary)
    summary = re.sub(r'입니다\s+입니다', '입니다', summary)
    
    # 3) 불필요한 진행형 단순화
    summary = re.sub(r'하고\s+있습니다', '합니다', summary)
    summary = re.sub(r'하고\s+있다', '한다', summary)
    summary = re.sub(r'하고\s+있으며', '하며', summary)
    summary = re.sub(r'하고\s+있고', '하고', summary)
    
    # 4) 중복 수식어 제거
    summary = re.sub(r'매우\s+많이', '많이', summary)
    summary = re.sub(r'정말\s+많이', '많이', summary)
    summary = re.sub(r'아주\s+많이', '많이', summary)
    summary = re.sub(r'너무\s+많이', '많이', summary)
    
    # 5) 반복되는 동사 형태 통일
    summary = re.sub(r'라고\s+말합니다', '라고 합니다', summary)
    summary = re.sub(r'이라고\s+말합니다', '이라고 합니다', summary)
    
    # 6) 불필요한 접속사 (문장 시작에서만)
    summary = re.sub(r'\.\s+그리고\s+', '. ', summary)
    summary = re.sub(r'\.\s+또한\s+', '. ', summary)
    
    # 7) 과도하게 긴 문장만 처리 (30단어 이상)
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    processed_sentences = []
    
    for sent in sentences:
        words = sent.split()
        if len(words) > 30:
            # 너무 긴 문장은 접속사로 분할 가능하면 분할
            if ' 그리고 ' in sent:
                parts = sent.split(' 그리고 ', 1)
                processed_sentences.extend(parts)
            elif ' 하지만 ' in sent:
                parts = sent.split(' 하지만 ', 1)
                processed_sentences.extend(parts)
            elif ', ' in sent and len(words) > 35:
                # 쉼표로 분할
                comma_idx = sent.rfind(',', 0, len(sent)//2)
                if comma_idx > 0:
                    processed_sentences.append(sent[:comma_idx+1])
                    processed_sentences.append(sent[comma_idx+1:].strip())
                else:
                    processed_sentences.append(sent)
            else:
                processed_sentences.append(sent)
        else:
            processed_sentences.append(sent)
    
    summary = ' '.join(processed_sentences)
    
    # 8) 불완전한 문장 끝 정리
    if summary and len(summary) > 10:
        # 마지막이 완전한 문장 종결어미인지 확인
        if not summary[-1] in '.!?':
            # 마지막 완전한 문장까지만
            last_period = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
            if last_period > len(summary) * 0.7:  # 70% 이후에 있으면
                summary = summary[:last_period+1]
    
    # 9) 최종 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'\s([,.!?])', r'\1', summary)
    summary = re.sub(r'\.\.+', '.', summary)  # 여러 개의 점 제거
    
    return summary

print(f"\n🔄 미세조정 적용 중...")

# 각 행에 대해 미세조정 적용
tuned_focused = []
for idx in range(len(v4)):
    original = v4.iloc[idx]['summary']
    tuned = focused_micro_tune_v4(original)
    tuned_focused.append(tuned)

# 통계 비교
v4_lengths = v4['summary'].apply(lambda x: len(str(x).split()))
tuned_lengths = [len(s.split()) for s in tuned_focused]

v4_mean = v4_lengths.mean()
tuned_mean = sum(tuned_lengths) / len(tuned_lengths)

print(f"\n📊 통계 비교:\n")
print(f"  원본 v4:")
print(f"    - 평균 길이: {v4_mean:.1f} 단어")

print(f"\n  집중 미세조정 (v2):")
print(f"    - 평균 길이: {tuned_mean:.1f} 단어 ({tuned_mean - v4_mean:+.1f})")

v3_lengths = v3_micro['summary'].apply(lambda x: len(str(x).split()))
print(f"\n  v3_microtuned (참고):")
print(f"    - 평균 길이: {v3_lengths.mean():.1f} 단어")

# 변화 케이스
changed = sum(1 for i in range(len(v4)) if v4.iloc[i]['summary'] != tuned_focused[i])
print(f"\n🔄 변화된 케이스: {changed}개 ({100*changed/len(v4):.1f}%)")

# 샘플 비교
print(f"\n" + "="*80)
print(f"🔍 주요 변화 샘플 (상위 3개)")
print(f"="*80)

changed_indices = [i for i in range(len(v4)) if v4.iloc[i]['summary'] != tuned_focused[i]]
for idx in changed_indices[:3]:
    orig = v4.iloc[idx]['summary']
    tuned = tuned_focused[idx]
    
    print(f"\n[{v4.iloc[idx]['fname']}]")
    print(f"  원본 ({len(orig.split())} 단어): {orig[:80]}...")
    print(f"  조정 ({len(tuned.split())} 단어): {tuned[:80]}...")

# 제출 파일 생성
output_path = './prediction/submit_solar_v4_focused_v2.csv'
submission = v4[['fname']].copy()
submission['summary'] = tuned_focused
submission.to_csv(output_path, index=False)

print(f"\n" + "="*80)
print(f"✅ 제출 파일 생성 완료")
print(f"="*80)

print(f"\n📁 파일: {output_path}")
print(f"📊 특징:")
print(f"  - 전략: 정보량 유지 + 품질 개선")
print(f"  - 평균 길이: {tuned_mean:.1f} 단어")
print(f"  - 변화율: {100*changed/len(v4):.1f}%")

print(f"\n🎯 다음 단계:")
print(f"  1. Dev 셋 평가로 성능 확인")
print(f"  2. v3_microtuned와 비교")
print(f"  3. 최종 제출 결정")

print(f"\n" + "="*80 + "\n")
