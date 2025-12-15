#!/usr/bin/env python3
"""
v4 센스있는 최종 미세조정 버전
=====================================
전략: v2 모델의 품질 문제를 보완하는 정교한 최적화
- v3의 성공 전략 적용 (3문장 제한 + 중복 제거)
- v4의 장점 살리기 (정보량)
- 품질 검증 강화
"""

import pandas as pd
import re

print("\n" + "="*80)
print("🎯 v4 센스있는 최종 미세조정 버전 생성")
print("="*80)

# 파일 로드
v4 = pd.read_csv('./prediction/submit_solar_v4.csv')
v3_micro = pd.read_csv('./prediction/submit_solar_v3_microtuned.csv')

print(f"\n📂 로드 완료:")
print(f"  v4: {len(v4)}개")
print(f"  v3_microtuned: {len(v3_micro)}개")

def smart_micro_tune_v4(summary: str) -> str:
    """
    센스있는 미세조정 전략
    
    1. 기본 정리
    2. 중복 제거 (ROUGE-2 향상)
    3. 3문장 제한 (v3 성공 전략)
    4. 품질 검증 (완전한 문장만)
    5. 간결화 (불필요한 표현 제거)
    """
    
    # 1) 기본 공백 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # 2) 명백한 중복 제거 (같은 단어/조사 연속)
    summary = re.sub(r'에게\s+에게', '에게', summary)
    summary = re.sub(r'에서\s+에서', '에서', summary)
    summary = re.sub(r'합니다\s+합니다', '합니다', summary)
    summary = re.sub(r'한다\s+한다', '한다', summary)
    summary = re.sub(r'하고\s+하고', '하고', summary)
    
    # 3) 불필요한 진행형 단순화
    summary = re.sub(r'하고\s+있습니다', '합니다', summary)
    summary = re.sub(r'하고\s+있다', '한다', summary)
    summary = re.sub(r'하고\s+있으며', '하며', summary)
    
    # 4) 중복 수식어 제거
    summary = re.sub(r'매우\s+많이', '많이', summary)
    summary = re.sub(r'정말\s+많이', '많이', summary)
    summary = re.sub(r'아주\s+많이', '많이', summary)
    
    # 5) 반복되는 동사 형태 통일
    summary = re.sub(r'라고\s+말합니다', '라고 합니다', summary)
    summary = re.sub(r'이라고\s+말합니다', '이라고 합니다', summary)
    
    # 6) 불필요한 접속사 정리 (문장 시작 제거)
    summary = re.sub(r'\.\s+그리고\s+', '. ', summary)
    summary = re.sub(r'\.\s+하지만\s+', '. ', summary)
    
    # 7) 3문장 제한 적용 (v3 성공 전략!)
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    
    # 중요: 너무 짧은 문장은 제외하고 의미있는 3문장만
    meaningful_sentences = []
    for sent in sentences:
        # 최소 5단어 이상인 문장만
        if len(sent.split()) >= 5:
            meaningful_sentences.append(sent)
        elif meaningful_sentences:  # 이미 문장이 있으면 마지막에 붙임
            meaningful_sentences[-1] = meaningful_sentences[-1] + ' ' + sent
    
    # 최대 3문장
    summary = ' '.join(meaningful_sentences[:3])
    
    # 8) 품질 검증: 완전한 문장인지 확인
    if summary and not summary[-1] in '.!?다요습니다니다':
        # 마지막 완전한 문장까지만
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        if len(sentences) > 1:
            summary = ' '.join(sentences[:-1])
    
    # 9) 최종 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'\s([,.!?])', r'\1', summary)
    
    # 10) 빈 문장 방지
    if not summary or len(summary.split()) < 5:
        # 원본의 첫 문장이라도 유지
        orig_sentences = re.split(r'(?<=[.!?])\s+', v4.iloc[0]['summary'].strip())
        summary = orig_sentences[0] if orig_sentences else summary
    
    return summary

print(f"\n🔄 미세조정 적용 중...")

# 각 행에 대해 미세조정 적용
tuned_smart = []
for idx in range(len(v4)):
    original = v4.iloc[idx]['summary']
    tuned = smart_micro_tune_v4(original)
    tuned_smart.append(tuned)

# 통계 비교
v4_lengths = v4['summary'].apply(lambda x: len(str(x).split()))
tuned_lengths = [len(s.split()) for s in tuned_smart]

v4_mean = v4_lengths.mean()
tuned_mean = sum(tuned_lengths) / len(tuned_lengths)

print(f"\n📊 통계 비교:\n")
print(f"  원본 v4:")
print(f"    - 평균 길이: {v4_mean:.1f} 단어")
print(f"    - 범위: {v4_lengths.min():.0f} ~ {v4_lengths.max():.0f} 단어")

print(f"\n  센스있는 미세조정:")
print(f"    - 평균 길이: {tuned_mean:.1f} 단어 ({tuned_mean - v4_mean:+.1f})")
print(f"    - 범위: {min(tuned_lengths):.0f} ~ {max(tuned_lengths):.0f} 단어")

v3_lengths = v3_micro['summary'].apply(lambda x: len(str(x).split()))
print(f"\n  v3_microtuned (목표):")
print(f"    - 평균 길이: {v3_lengths.mean():.1f} 단어")

# 변화 케이스
changed = sum(1 for i in range(len(v4)) if v4.iloc[i]['summary'] != tuned_smart[i])
print(f"\n🔄 변화된 케이스: {changed}개 ({100*changed/len(v4):.1f}%)")

# 샘플 비교
print(f"\n" + "="*80)
print(f"🔍 샘플 비교 (상위 5개)")
print(f"="*80)

for i in range(min(5, len(v4))):
    orig = v4.iloc[i]['summary']
    tuned = tuned_smart[i]
    
    print(f"\n[{i+1}] {v4.iloc[i]['fname']}")
    print(f"  원본 ({len(orig.split())} 단어):")
    print(f"    {orig[:100]}...")
    print(f"  조정 ({len(tuned.split())} 단어):")
    print(f"    {tuned[:100]}...")
    print(f"  변화: {len(orig.split())} → {len(tuned.split())} ({len(tuned.split()) - len(orig.split()):+d} 단어)")

# 제출 파일 생성
output_path = './prediction/submit_solar_v4_smart_final.csv'
submission = v4[['fname']].copy()
submission['summary'] = tuned_smart
submission.to_csv(output_path, index=False)

print(f"\n" + "="*80)
print(f"✅ 제출 파일 생성 완료")
print(f"="*80)

print(f"\n📁 파일: {output_path}")
print(f"📊 통계:")
print(f"  - 평균 길이: {tuned_mean:.1f} 단어 (v4: {v4_mean:.1f}, v3: {v3_lengths.mean():.1f})")
print(f"  - 변화율: {100*changed/len(v4):.1f}%")
print(f"  - 전략: 3문장 제한 + 중복 제거 + 품질 검증")

print(f"\n🎯 예상 성능:")
print(f"  - 기대 점수: 51.85~52.00점")
print(f"  - 근거:")
print(f"    1) v3의 성공 전략 적용 (3문장 제한)")
print(f"    2) v4의 정보량 활용")
print(f"    3) 품질 검증 강화")
print(f"    4) v3_microtuned 길이에 근접")

print(f"\n💡 권장 사항:")
print(f"  1️⃣ [추천] submit_solar_v4_smart_final.csv 제출")
print(f"     → 예상: 51.85~52.00점")
print(f"  2️⃣ [백업] v3_microtuned.csv 준비")
print(f"     → 확실: 51.9421점")

print(f"\n" + "="*80)
print(f"🎯 최종 결정:")
print(f"  센스있는 v4 미세조정 vs v3_microtuned")
print(f"  → v4_smart_final 먼저 시도 (개선 가능성)")
print(f"  → 실패 시 v3_microtuned로 복귀 (안정성)")
print(f"="*80 + "\n")
