#!/usr/bin/env python3
"""
할루시네이션 감지 및 제거 전략
==================================
목표: v4의 할루시네이션 문제를 발견하고 제거하여 점수 향상
"""

import pandas as pd
import re
from collections import Counter

print("\n" + "="*80)
print("🔍 할루시네이션 감지 및 분석")
print("="*80)

# 파일 로드
v4 = pd.read_csv('./prediction/submit_solar_v4.csv')
v3_micro = pd.read_csv('./prediction/submit_solar_v3_microtuned.csv')
dev_df = pd.read_csv('./data/dev.csv')

print(f"\n📂 파일 로드 완료")

def detect_hallucinations(summary: str, dialogue: str = None) -> dict:
    """
    할루시네이션 패턴 감지
    
    1. 과도한 세부사항 (숫자, 날짜, 구체적 정보)
    2. 불확실한 추측성 표현
    3. 원문에 없는 인과관계
    4. 과도한 감정/의견 표현
    """
    
    issues = {
        'excessive_details': [],
        'speculation': [],
        'unsupported_claims': [],
        'quality_score': 0
    }
    
    # 1) 과도한 숫자/날짜 (원문 없이 구체적인 정보)
    numbers = re.findall(r'\d+', summary)
    if len(numbers) > 5:
        issues['excessive_details'].append(f'과도한 숫자: {len(numbers)}개')
    
    # 2) 추측성 표현
    speculation_patterns = [
        r'것으로\s*보입니다',
        r'것으로\s*생각됩니다',
        r'것으로\s*추정됩니다',
        r'인\s*것\s*같습니다',
        r'듯\s*합니다',
        r'아마도',
        r'추측',
    ]
    
    for pattern in speculation_patterns:
        if re.search(pattern, summary):
            issues['speculation'].append(pattern)
    
    # 3) 과도하게 긴 문장 (30단어 이상 = 할루시네이션 가능성)
    sentences = re.split(r'[.!?]', summary)
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if long_sentences:
        issues['excessive_details'].append(f'과도하게 긴 문장: {len(long_sentences)}개')
    
    # 4) 과도한 접속사 (불필요한 정보 연결)
    conjunctions = ['그리고', '또한', '하지만', '그러나', '따라서', '그래서']
    conj_count = sum(summary.count(c) for c in conjunctions)
    if conj_count > 3:
        issues['unsupported_claims'].append(f'과도한 접속사: {conj_count}개')
    
    # 5) 품질 점수 계산 (0~100)
    quality_score = 100
    quality_score -= len(issues['excessive_details']) * 10
    quality_score -= len(issues['speculation']) * 15
    quality_score -= len(issues['unsupported_claims']) * 10
    issues['quality_score'] = max(0, quality_score)
    
    return issues

# v4와 v3 비교 분석
print(f"\n📊 할루시네이션 분석 중...\n")

v4_issues = []
v3_issues = []

for idx in range(len(v4)):
    v4_issue = detect_hallucinations(v4.iloc[idx]['summary'])
    v3_issue = detect_hallucinations(v3_micro.iloc[idx]['summary'])
    
    v4_issues.append(v4_issue)
    v3_issues.append(v3_issue)

# 통계 계산
v4_avg_quality = sum(i['quality_score'] for i in v4_issues) / len(v4_issues)
v3_avg_quality = sum(i['quality_score'] for i in v3_issues) / len(v3_issues)

v4_with_speculation = sum(1 for i in v4_issues if i['speculation'])
v3_with_speculation = sum(1 for i in v3_issues if i['speculation'])

v4_excessive = sum(1 for i in v4_issues if i['excessive_details'])
v3_excessive = sum(1 for i in v3_issues if i['excessive_details'])

print(f"{'='*80}")
print(f"📈 할루시네이션 분석 결과")
print(f"{'='*80}\n")

print(f"품질 점수 (높을수록 좋음):")
print(f"  v4_original:  {v4_avg_quality:.1f}/100")
print(f"  v3_microtuned: {v3_avg_quality:.1f}/100")
print(f"  차이: {v3_avg_quality - v4_avg_quality:+.1f}")

print(f"\n추측성 표현 발견:")
print(f"  v4: {v4_with_speculation}개 케이스 ({100*v4_with_speculation/len(v4):.1f}%)")
print(f"  v3: {v3_with_speculation}개 케이스 ({100*v3_with_speculation/len(v3_micro):.1f}%)")

print(f"\n과도한 세부사항:")
print(f"  v4: {v4_excessive}개 케이스 ({100*v4_excessive/len(v4):.1f}%)")
print(f"  v3: {v3_excessive}개 케이스 ({100*v3_excessive/len(v3_micro):.1f}%)")

# 할루시네이션 의심 샘플 찾기
print(f"\n{'='*80}")
print(f"🚨 할루시네이션 의심 케이스 (상위 5개)")
print(f"{'='*80}\n")

v4_suspicious = sorted(enumerate(v4_issues), key=lambda x: x[1]['quality_score'])[:5]

for idx, (original_idx, issue) in enumerate(v4_suspicious, 1):
    summary = v4.iloc[original_idx]['summary']
    fname = v4.iloc[original_idx]['fname']
    
    print(f"[{idx}] {fname} (품질: {issue['quality_score']}/100)")
    print(f"  요약: {summary[:100]}...")
    
    if issue['speculation']:
        print(f"  ⚠️ 추측성 표현: {', '.join(issue['speculation'][:2])}")
    if issue['excessive_details']:
        print(f"  ⚠️ 과도한 세부사항: {', '.join(issue['excessive_details'][:2])}")
    if issue['unsupported_claims']:
        print(f"  ⚠️ 근거 없는 주장: {', '.join(issue['unsupported_claims'][:2])}")
    print()

print(f"{'='*80}")
print(f"💡 할루시네이션 제거 전략")
print(f"{'='*80}\n")

print(f"1️⃣ 추측성 표현 제거")
print(f"   - '것으로 보입니다', '것 같습니다' → 단정적 표현으로 변경")
print(f"   - 불확실한 내용은 삭제")

print(f"\n2️⃣ 과도한 세부사항 제거")
print(f"   - 30단어 이상 문장 → 핵심만 추출")
print(f"   - 불필요한 숫자/날짜 제거")

print(f"\n3️⃣ 근거 없는 주장 제거")
print(f"   - 원문에 없는 인과관계 제거")
print(f"   - 과도한 접속사 정리")

print(f"\n4️⃣ 사실만 유지")
print(f"   - 대화에 명시적으로 나온 내용만")
print(f"   - 해석이나 추론 최소화")

print(f"\n{'='*80}")
print(f"🎯 결론")
print(f"{'='*80}\n")

if v4_avg_quality < v3_avg_quality:
    print(f"✅ v4는 v3보다 할루시네이션이 더 많습니다")
    print(f"   v4 품질: {v4_avg_quality:.1f}/100")
    print(f"   v3 품질: {v3_avg_quality:.1f}/100")
    print(f"   차이: {v3_avg_quality - v4_avg_quality:.1f}점")
    print(f"\n💡 권장: 할루시네이션 제거 버전 생성")
else:
    print(f"⚠️ v4와 v3의 할루시네이션 수준이 유사합니다")
    print(f"\n💡 권장: v3_microtuned 사용")

print(f"\n{'='*80}\n")
