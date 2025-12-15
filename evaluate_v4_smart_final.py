#!/usr/bin/env python3
"""
v4_smart_final 성능 검증
"""

import pandas as pd
from evaluate import load

print("\n" + "="*80)
print("📊 v4_smart_final 성능 검증 (Dev 셋 ROUGE 평가)")
print("="*80)

# 파일 로드
v4_smart = pd.read_csv('./prediction/submit_solar_v4_smart_final.csv')
v4_original = pd.read_csv('./prediction/submit_solar_v4.csv')
v3_micro = pd.read_csv('./prediction/submit_solar_v3_microtuned.csv')
dev_df = pd.read_csv('./data/dev.csv')

print(f"\n📂 로드 완료:")
print(f"  v4_smart_final: {len(v4_smart)}개")
print(f"  v4_original: {len(v4_original)}개")
print(f"  v3_microtuned: {len(v3_micro)}개")
print(f"  dev: {len(dev_df)}개")

# ROUGE 평가
rouge = load("rouge")

print(f"\n⏳ ROUGE 평가 중...\n")

versions = {
    'v4_smart_final': v4_smart['summary'].tolist(),
    'v4_original': v4_original['summary'].tolist(),
    'v3_microtuned': v3_micro['summary'].tolist(),
}

results = {}
for name, summaries in versions.items():
    print(f"  평가 중: {name:20s}", end='', flush=True)
    scores = rouge.compute(predictions=summaries, references=dev_df['summary'].tolist())
    results[name] = {
        'R1': scores['rouge1'] * 100,
        'R2': scores['rouge2'] * 100,
        'RL': scores['rougeL'] * 100,
        'Combined': (scores['rouge1'] + scores['rouge2'] + scores['rougeL']) / 3 * 100
    }
    print(" ✅")

# 결과 정리
print(f"\n" + "="*80)
print(f"📈 ROUGE 평가 결과")
print(f"="*80 + "\n")

results_df = pd.DataFrame([
    {
        'Version': k,
        'R1': v['R1'],
        'R2': v['R2'],
        'RL': v['RL'],
        'Combined': v['Combined']
    }
    for k, v in sorted(results.items(), key=lambda x: x[1]['Combined'], reverse=True)
])

print(results_df.to_string(index=False))

# 상세 분석
print(f"\n" + "="*80)
print(f"🎯 결과 분석")
print(f"="*80)

smart_r2 = results['v4_smart_final']['R2']
smart_combined = results['v4_smart_final']['Combined']

v4_orig_r2 = results['v4_original']['R2']
v4_orig_combined = results['v4_original']['Combined']

v3_r2 = results['v3_microtuned']['R2']
v3_combined = results['v3_microtuned']['Combined']

print(f"\n✅ v4_smart_final 성능:")
print(f"   ROUGE-2: {smart_r2:.2f}%")
print(f"   Combined: {smart_combined:.4f}")

print(f"\n📊 비교:")
print(f"   vs v4_original:")
print(f"     ROUGE-2: {smart_r2 - v4_orig_r2:+.2f}%p")
print(f"     Combined: {smart_combined - v4_orig_combined:+.4f}")

print(f"\n   vs v3_microtuned (51.9421점):")
print(f"     ROUGE-2: {smart_r2 - v3_r2:+.2f}%p")
print(f"     Combined: {smart_combined - v3_combined:+.4f}")

# 리더보드 점수 예측
print(f"\n" + "="*80)
print(f"🎲 리더보드 점수 예측")
print(f"="*80)

# v3의 dev ROUGE와 리더보드 점수 관계 활용
# v3: Combined 36.97 → 리더보드 51.9421
# 비율: 51.9421 / 36.97 ≈ 1.404

ratio = 51.9421 / v3_combined
predicted_score = smart_combined * ratio

print(f"\n📈 예측 모델:")
print(f"   v3_microtuned: Dev Combined {v3_combined:.4f} → 리더보드 51.9421")
print(f"   변환 비율: {ratio:.4f}")

print(f"\n🎯 v4_smart_final 예측:")
print(f"   Dev Combined: {smart_combined:.4f}")
print(f"   예상 리더보드 점수: {predicted_score:.4f}")

# 보수적/낙관적 시나리오
conservative = predicted_score - 0.05
optimistic = predicted_score + 0.05

print(f"\n   보수적 시나리오: {conservative:.4f}")
print(f"   예상값: {predicted_score:.4f}")
print(f"   낙관적 시나리오: {optimistic:.4f}")

# 최종 권장
print(f"\n" + "="*80)
print(f"💡 최종 권장사항")
print(f"="*80)

if smart_combined > v3_combined:
    print(f"\n✅ v4_smart_final이 v3_microtuned보다 우수합니다!")
    print(f"   → v4_smart_final 제출 권장")
    print(f"   → 예상 점수: {predicted_score:.4f} (v3: 51.9421보다 높음)")
elif smart_combined > v4_orig_combined:
    print(f"\n🔄 v4_smart_final이 v4_original보다 개선되었습니다!")
    print(f"   → v4_smart_final vs v3_microtuned 비교:")
    print(f"     • v4_smart: 예상 {predicted_score:.4f}")
    print(f"     • v3_micro: 확실 51.9421")
    
    if predicted_score > 51.94:
        print(f"\n   ✅ v4_smart_final 제출 권장 (예상값이 v3보다 높음)")
    else:
        print(f"\n   ⚠️ v3_microtuned 제출 권장 (확실성 우선)")
else:
    print(f"\n⚠️ v4_smart_final이 v4_original과 유사하거나 낮습니다")
    print(f"   → v3_microtuned 제출 권장 (51.9421 보장)")

print(f"\n" + "="*80 + "\n")
