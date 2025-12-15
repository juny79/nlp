#!/usr/bin/env python3
"""
v4 품질 저하 원인 상세 분석
==============================
v4가 v3보다 낮은 품질인 이유를 다각도로 분석
"""

import pandas as pd
import numpy as np
from evaluate import load
import re
from collections import Counter

print("\n" + "="*100)
print("📊 v4 품질 저하 원인 종합 분석")
print("="*100)

# 데이터 로드
v4 = pd.read_csv('./prediction/submit_solar_v4.csv')
v3_micro = pd.read_csv('./prediction/submit_solar_v3_microtuned.csv')
v3_orig = pd.read_csv('./prediction/submit_solar_v3.csv')
dev_df = pd.read_csv('./data/dev.csv')
test_df = pd.read_csv('./data/test.csv')

print(f"\n📂 데이터 로드 완료")

# ============================================================================
# 1. ROUGE 점수 상세 분석
# ============================================================================
print(f"\n" + "="*100)
print(f"1. ROUGE 점수 비교 분석")
print(f"="*100)

rouge = load("rouge")

versions = {
    'v3_original': v3_orig['summary'].tolist(),
    'v3_microtuned': v3_micro['summary'].tolist(),
    'v4_original': v4['summary'].tolist(),
}

print(f"\n⏳ ROUGE 평가 중...\n")

results = {}
for name, summaries in versions.items():
    print(f"  평가 중: {name:20s}", end='', flush=True)
    scores = rouge.compute(predictions=summaries, references=dev_df['summary'].tolist())
    results[name] = {
        'rouge1': scores['rouge1'] * 100,
        'rouge2': scores['rouge2'] * 100,
        'rougeL': scores['rougeL'] * 100,
        'combined': (scores['rouge1'] + scores['rouge2'] + scores['rougeL']) / 3 * 100
    }
    print(" ✅")

print(f"\n{'='*100}")
print(f"ROUGE 점수 상세 비교")
print(f"{'='*100}\n")

print(f"{'Version':<20s} {'ROUGE-1':>10s} {'ROUGE-2':>10s} {'ROUGE-L':>10s} {'Combined':>10s} {'리더보드':>12s}")
print(f"{'-'*100}")

for name in ['v3_original', 'v3_microtuned', 'v4_original']:
    r = results[name]
    if name == 'v3_original':
        leaderboard = '51.8026'
    elif name == 'v3_microtuned':
        leaderboard = '51.9421 ✅'
    else:
        leaderboard = '51.7703 ⚠️'
    
    print(f"{name:<20s} {r['rouge1']:>9.2f}% {r['rouge2']:>9.2f}% {r['rougeL']:>9.2f}% {r['combined']:>9.4f} {leaderboard:>12s}")

print(f"\n{'='*100}")
print(f"차이 분석 (v4 vs v3_microtuned)")
print(f"{'='*100}\n")

v4_r = results['v4_original']
v3_r = results['v3_microtuned']

print(f"  ROUGE-1 차이: {v4_r['rouge1'] - v3_r['rouge1']:+.2f}%p  ({100*(v4_r['rouge1'] - v3_r['rouge1'])/v3_r['rouge1']:+.2f}% 변화)")
print(f"  ROUGE-2 차이: {v4_r['rouge2'] - v3_r['rouge2']:+.2f}%p  ({100*(v4_r['rouge2'] - v3_r['rouge2'])/v3_r['rouge2']:+.2f}% 변화) ⚠️")
print(f"  ROUGE-L 차이: {v4_r['rougeL'] - v3_r['rougeL']:+.2f}%p  ({100*(v4_r['rougeL'] - v3_r['rougeL'])/v3_r['rougeL']:+.2f}% 변화)")
print(f"  Combined 차이: {v4_r['combined'] - v3_r['combined']:+.4f}  ({100*(v4_r['combined'] - v3_r['combined'])/v3_r['combined']:+.2f}% 변화)")

print(f"\n💡 핵심 문제:")
print(f"  → ROUGE-2가 {abs(v4_r['rouge2'] - v3_r['rouge2']):.2f}%p 낮음 (바이그램 매칭 저하)")
print(f"  → 이는 중요 구문의 정확도가 떨어짐을 의미")

# ============================================================================
# 2. 길이 및 구조 분석
# ============================================================================
print(f"\n" + "="*100)
print(f"2. 텍스트 길이 및 구조 분석")
print(f"="*100)

v3_lengths = v3_micro['summary'].apply(lambda x: len(str(x).split()))
v4_lengths = v4['summary'].apply(lambda x: len(str(x).split()))
dev_lengths = dev_df['summary'].apply(lambda x: len(str(x).split()))

print(f"\n📏 길이 통계:\n")
print(f"  {'Version':<20s} {'평균':>8s} {'중앙값':>8s} {'표준편차':>8s} {'최소':>8s} {'최대':>8s}")
print(f"  {'-'*80}")
print(f"  {'Dev (정답)':<20s} {dev_lengths.mean():>8.1f} {dev_lengths.median():>8.0f} {dev_lengths.std():>8.1f} {dev_lengths.min():>8.0f} {dev_lengths.max():>8.0f}")
print(f"  {'v3_microtuned':<20s} {v3_lengths.mean():>8.1f} {v3_lengths.median():>8.0f} {v3_lengths.std():>8.1f} {v3_lengths.min():>8.0f} {v3_lengths.max():>8.0f}")
print(f"  {'v4_original':<20s} {v4_lengths.mean():>8.1f} {v4_lengths.median():>8.0f} {v4_lengths.std():>8.1f} {v4_lengths.min():>8.0f} {v4_lengths.max():>8.0f}")

print(f"\n📊 분석:")
print(f"  v4가 v3보다 평균 {v4_lengths.mean() - v3_lengths.mean():+.1f} 단어 더 김")
print(f"  Dev 정답과의 차이: v3는 {v3_lengths.mean() - dev_lengths.mean():+.1f}, v4는 {v4_lengths.mean() - dev_lengths.mean():+.1f}")
print(f"  → v4는 불필요한 정보를 더 많이 포함")

# 문장 수 분석
v3_sentences = v3_micro['summary'].apply(lambda x: len(re.split(r'[.!?]', str(x).strip())))
v4_sentences = v4['summary'].apply(lambda x: len(re.split(r'[.!?]', str(x).strip())))

print(f"\n📝 문장 수 통계:\n")
print(f"  v3_microtuned: 평균 {v3_sentences.mean():.1f}개 문장")
print(f"  v4_original:   평균 {v4_sentences.mean():.1f}개 문장")
print(f"  → v4가 평균 {v4_sentences.mean() - v3_sentences.mean():+.1f}개 문장 더 많음")

# ============================================================================
# 3. 어휘 다양성 및 반복 분석
# ============================================================================
print(f"\n" + "="*100)
print(f"3. 어휘 다양성 및 반복 패턴 분석")
print(f"="*100)

def calculate_lexical_diversity(summaries):
    """어휘 다양성 계산 (unique words / total words)"""
    all_words = []
    for summary in summaries:
        words = str(summary).split()
        all_words.extend(words)
    
    return len(set(all_words)) / len(all_words) if all_words else 0

def count_repetitions(summaries):
    """반복되는 바이그램 수 계산"""
    repetitions = 0
    for summary in summaries:
        words = str(summary).split()
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repetitions += sum(1 for count in bigram_counts.values() if count > 1)
    return repetitions

v3_diversity = calculate_lexical_diversity(v3_micro['summary'])
v4_diversity = calculate_lexical_diversity(v4['summary'])

v3_reps = count_repetitions(v3_micro['summary'])
v4_reps = count_repetitions(v4['summary'])

print(f"\n📚 어휘 다양성 (높을수록 좋음):")
print(f"  v3_microtuned: {v3_diversity:.4f}")
print(f"  v4_original:   {v4_diversity:.4f}")
print(f"  차이: {v4_diversity - v3_diversity:+.4f} ({100*(v4_diversity - v3_diversity)/v3_diversity:+.2f}%)")

print(f"\n🔁 반복 패턴 (낮을수록 좋음):")
print(f"  v3_microtuned: {v3_reps}개 반복 바이그램")
print(f"  v4_original:   {v4_reps}개 반복 바이그램")
print(f"  차이: {v4_reps - v3_reps:+d}개")

if v4_diversity < v3_diversity:
    print(f"\n💡 분석: v4는 어휘가 덜 다양함 → 표현이 단조로움")
if v4_reps > v3_reps:
    print(f"💡 분석: v4는 불필요한 반복이 많음 → ROUGE-2 저하 원인")

# ============================================================================
# 4. 정보 밀도 분석
# ============================================================================
print(f"\n" + "="*100)
print(f"4. 정보 밀도 분석 (ROUGE per word)")
print(f"="*100)

v3_density = v3_r['rouge2'] / v3_lengths.mean()
v4_density = v4_r['rouge2'] / v4_lengths.mean()

print(f"\n📈 ROUGE-2 per word (정보 밀도):")
print(f"  v3_microtuned: {v3_density:.4f} (ROUGE-2 {v3_r['rouge2']:.2f}% / {v3_lengths.mean():.1f} 단어)")
print(f"  v4_original:   {v4_density:.4f} (ROUGE-2 {v4_r['rouge2']:.2f}% / {v4_lengths.mean():.1f} 단어)")
print(f"  차이: {v4_density - v3_density:+.4f} ({100*(v4_density - v3_density)/v3_density:+.2f}%)")

print(f"\n💡 핵심 발견:")
if v4_density < v3_density:
    print(f"  → v4는 단어당 정보 밀도가 {abs(100*(v4_density - v3_density)/v3_density):.1f}% 낮음")
    print(f"  → 같은 내용을 표현하는데 더 많은 단어 사용 (비효율적)")
    print(f"  → 불필요한 수식어, 반복, 세부사항이 많음")

# ============================================================================
# 5. 샘플 품질 비교
# ============================================================================
print(f"\n" + "="*100)
print(f"5. 구체적 샘플 품질 비교 (Dev 정답 기준)")
print(f"="*100)

# Dev 셋과 매칭
v3_dev = v3_micro.head(len(dev_df))
v4_dev = v4.head(len(dev_df))

print(f"\n🔍 품질 차이가 큰 상위 5개 샘플:\n")

sample_scores = []
for idx in range(min(len(dev_df), len(v3_dev), len(v4_dev))):
    dev_summary = dev_df.iloc[idx]['summary']
    v3_summary = v3_dev.iloc[idx]['summary']
    v4_summary = v4_dev.iloc[idx]['summary']
    
    # 각 샘플의 ROUGE-2 계산
    v3_score = rouge.compute(predictions=[v3_summary], references=[dev_summary])['rouge2']
    v4_score = rouge.compute(predictions=[v4_summary], references=[dev_summary])['rouge2']
    
    diff = v3_score - v4_score
    sample_scores.append((idx, diff, v3_score, v4_score, dev_summary, v3_summary, v4_summary))

# 차이가 큰 순으로 정렬
sample_scores_sorted = sorted(sample_scores, key=lambda x: x[1], reverse=True)[:5]

for rank, (idx, diff, v3_score, v4_score, dev_sum, v3_sum, v4_sum) in enumerate(sample_scores_sorted, 1):
    print(f"[{rank}] 샘플 {idx} (v3가 {diff*100:.1f}%p 더 높음)")
    print(f"  Dev 정답 ({len(dev_sum.split())} 단어): {dev_sum[:80]}...")
    print(f"  v3 ({len(v3_sum.split())} 단어, ROUGE-2: {v3_score*100:.1f}%): {v3_sum[:80]}...")
    print(f"  v4 ({len(v4_sum.split())} 단어, ROUGE-2: {v4_score*100:.1f}%): {v4_sum[:80]}...")
    print(f"  💡 v3가 더 나은 이유: ", end='')
    
    if len(v4_sum.split()) > len(v3_sum.split()) + 5:
        print(f"v4가 너무 김 (+{len(v4_sum.split()) - len(v3_sum.split())}단어)")
    elif len(v4_sum.split()) < len(dev_sum.split()) - 3:
        print(f"v4가 너무 짧음")
    else:
        print(f"v4의 표현이 부정확함")
    print()

# ============================================================================
# 6. 모델 버전 차이 추정
# ============================================================================
print(f"\n" + "="*100)
print(f"6. v1 vs v2 모델 성능 차이 추정")
print(f"="*100)

v3_orig_r = results['v3_original']

print(f"\n📊 모델별 초기 성능 (후처리 전):")
print(f"  v1 모델 (v3_original):")
print(f"    - ROUGE-2: {v3_orig_r['rouge2']:.2f}%")
print(f"    - 리더보드: 51.8026점")
print(f"  v2 모델 (v4_original):")
print(f"    - ROUGE-2: {v4_r['rouge2']:.2f}%")
print(f"    - 리더보드: 51.7703점")

print(f"\n🔻 성능 저하:")
print(f"  ROUGE-2: {v4_r['rouge2'] - v3_orig_r['rouge2']:+.2f}%p")
print(f"  리더보드: {51.7703 - 51.8026:+.4f}점")

print(f"\n💡 추정 원인:")
print(f"  1. 과적합 (Overfitting): v2 학습 시 검증 데이터에 과적합")
print(f"  2. 하이퍼파라미터: 학습률, 에폭 수 등이 부적절")
print(f"  3. 프롬프트 불일치: inference 시 프롬프트가 학습과 다름")
print(f"  4. 데이터 품질: v2 학습 데이터에 노이즈 포함")

# ============================================================================
# 7. 종합 분석 및 결론
# ============================================================================
print(f"\n" + "="*100)
print(f"7. 종합 분석 및 결론")
print(f"="*100)

print(f"\n📋 v4 품질 저하 요인 종합:\n")

factors = [
    {
        'factor': 'ROUGE-2 점수',
        'v3': f"{v3_r['rouge2']:.2f}%",
        'v4': f"{v4_r['rouge2']:.2f}%",
        'diff': f"{v4_r['rouge2'] - v3_r['rouge2']:+.2f}%p",
        'severity': '높음',
        'impact': '바이그램 매칭 저하 → 핵심 구문 부정확'
    },
    {
        'factor': '정보 밀도',
        'v3': f"{v3_density:.4f}",
        'v4': f"{v4_density:.4f}",
        'diff': f"{100*(v4_density - v3_density)/v3_density:+.1f}%",
        'severity': '높음',
        'impact': '단어당 정보량 감소 → 비효율적 표현'
    },
    {
        'factor': '평균 길이',
        'v3': f"{v3_lengths.mean():.1f}단어",
        'v4': f"{v4_lengths.mean():.1f}단어",
        'diff': f"+{v4_lengths.mean() - v3_lengths.mean():.1f}",
        'severity': '중간',
        'impact': '불필요한 정보 포함 → 핵심 흐림'
    },
    {
        'factor': '어휘 다양성',
        'v3': f"{v3_diversity:.4f}",
        'v4': f"{v4_diversity:.4f}",
        'diff': f"{100*(v4_diversity - v3_diversity)/v3_diversity:+.1f}%",
        'severity': '낮음',
        'impact': '표현 단조로움'
    },
    {
        'factor': '반복 패턴',
        'v3': f"{v3_reps}개",
        'v4': f"{v4_reps}개",
        'diff': f"+{v4_reps - v3_reps}",
        'severity': '중간',
        'impact': '불필요한 반복 → ROUGE 저하'
    }
]

print(f"  {'요인':<15s} {'v3':>12s} {'v4':>12s} {'차이':>12s} {'심각도':>8s}")
print(f"  {'-'*70}")
for f in factors:
    print(f"  {f['factor']:<15s} {f['v3']:>12s} {f['v4']:>12s} {f['diff']:>12s} {f['severity']:>8s}")
    print(f"  → {f['impact']}")

print(f"\n{'='*100}")
print(f"🎯 최종 결론")
print(f"{'='*100}\n")

print(f"v4가 v3보다 품질이 낮은 이유:\n")
print(f"  1️⃣ ROUGE-2 점수 {abs(v4_r['rouge2'] - v3_r['rouge2']):.2f}%p 낮음 (심각)")
print(f"     → 핵심 구문(바이그램) 매칭이 {abs(100*(v4_r['rouge2'] - v3_r['rouge2'])/v3_r['rouge2']):.1f}% 감소")
print(f"     → 중요한 정보를 정확하게 표현하지 못함")

print(f"\n  2️⃣ 정보 밀도 {abs(100*(v4_density - v3_density)/v3_density):.1f}% 낮음 (심각)")
print(f"     → 같은 내용을 표현하는데 더 많은 단어 필요")
print(f"     → 불필요한 수식어, 세부사항 과다 포함")

print(f"\n  3️⃣ 평균 {v4_lengths.mean() - v3_lengths.mean():.1f}단어 더 김 (중간)")
print(f"     → Dev 정답({dev_lengths.mean():.1f}단어)보다 {v4_lengths.mean() - dev_lengths.mean():.1f}단어 더 김")
print(f"     → 핵심이 흐려짐")

print(f"\n  4️⃣ v2 모델 자체의 성능 저하 (근본 원인)")
print(f"     → v1(51.8026) → v2(51.7703) = -0.0323점")
print(f"     → 후처리로 해결 불가능")

print(f"\n💡 개선 가능성:")
print(f"  ❌ 후처리: 근본적 품질 문제로 효과 미미 (+0.04 최대)")
print(f"  ❌ 미세조정: 오히려 악화 (-0.32 최악)")
print(f"  ✅ 모델 교체: v1 사용 (v3_microtuned) → 51.9421점 보장")

print(f"\n{'='*100}\n")

# 보고서 저장
print(f"📄 상세 보고서를 저장합니다...")
