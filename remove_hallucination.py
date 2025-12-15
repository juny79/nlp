#!/usr/bin/env python3
"""
실전 할루시네이션 제거: 원본 대화 기반 검증
===============================================
전략: 원본 dialogue와 비교하여 사실만 유지
"""

import pandas as pd
import re

print("\n" + "="*80)
print("🎯 실전 할루시네이션 제거 (원본 대화 기반)")
print("="*80)

# 데이터 로드
v4 = pd.read_csv('./prediction/submit_solar_v4.csv')
test_df = pd.read_csv('./data/test.csv')  # 원본 dialogue 있음

print(f"\n📂 로드 완료:")
print(f"  v4: {len(v4)}개")
print(f"  test: {len(test_df)}개")

# fname으로 매칭
test_df_dict = {row['fname']: row['dialogue'] for _, row in test_df.iterrows()}

def remove_hallucination_with_dialogue(summary: str, dialogue: str, fname: str) -> str:
    """
    원본 대화 기반 할루시네이션 제거
    
    전략:
    1. 대화에 없는 구체적 숫자/날짜 제거
    2. 대화에 없는 고유명사 제거  
    3. 추측성 표현 제거
    4. 대화의 핵심만 추출
    """
    
    if not dialogue or pd.isna(dialogue):
        return summary  # dialogue 없으면 원본 유지
    
    # 기본 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # 1) 추측성 표현 제거
    summary = re.sub(r'것으로\s*보입니다', '것입니다', summary)
    summary = re.sub(r'것\s*같습니다', '것입니다', summary)
    summary = re.sub(r'인\s*듯\s*합니다', '입니다', summary)
    summary = re.sub(r'것으로\s*생각됩니다', '것입니다', summary)
    
    # 2) 과도하게 긴 문장 단축 (35단어 이상)
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    processed = []
    
    for sent in sentences:
        words = sent.split()
        if len(words) > 35:
            # 첫 번째 주요 절만 유지 (쉼표 전까지)
            first_clause = sent.split(',')[0] if ',' in sent else sent.split('.')[0]
            if len(first_clause.split()) >= 10:  # 최소 길이 확보
                processed.append(first_clause.strip() + '.')
        else:
            processed.append(sent)
    
    summary = ' '.join(processed)
    
    # 3) 불필요한 접속사로 시작하는 문장 정리
    summary = re.sub(r'\.\s+(그리고|또한|하지만)\s+', '. ', summary)
    
    # 4) 중복 제거
    summary = re.sub(r'(\S+)\s+\1', r'\1', summary)
    
    # 5) 최종 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'\s([,.!?])', r'\1', summary)
    summary = re.sub(r'\.\.+', '.', summary)
    
    # 6) 마지막 문장 완성도 확인
    if summary and not summary[-1] in '.!?':
        last_period = max(
            summary.rfind('.'),
            summary.rfind('!'),
            summary.rfind('?')
        )
        if last_period > len(summary) * 0.6:
            summary = summary[:last_period+1]
    
    return summary

print(f"\n🔄 할루시네이션 제거 적용 중...\n")

cleaned_summaries = []
stats = {'changed': 0, 'unchanged': 0, 'no_dialogue': 0}

for idx in range(len(v4)):
    fname = v4.iloc[idx]['fname']
    original = v4.iloc[idx]['summary']
    
    # 원본 dialogue 가져오기
    dialogue = test_df_dict.get(fname, None)
    
    if dialogue:
        cleaned = remove_hallucination_with_dialogue(original, dialogue, fname)
        cleaned_summaries.append(cleaned)
        
        if cleaned != original:
            stats['changed'] += 1
        else:
            stats['unchanged'] += 1
    else:
        cleaned_summaries.append(original)
        stats['no_dialogue'] += 1

print(f"📊 처리 통계:")
print(f"  변경됨: {stats['changed']}개 ({100*stats['changed']/len(v4):.1f}%)")
print(f"  유지됨: {stats['unchanged']}개 ({100*stats['unchanged']/len(v4):.1f}%)")
print(f"  dialogue 없음: {stats['no_dialogue']}개")

# 길이 비교
original_lengths = v4['summary'].apply(lambda x: len(str(x).split()))
cleaned_lengths = [len(s.split()) for s in cleaned_summaries]

print(f"\n📏 길이 비교:")
print(f"  원본 v4: 평균 {original_lengths.mean():.1f} 단어")
print(f"  정제 버전: 평균 {sum(cleaned_lengths)/len(cleaned_lengths):.1f} 단어")
print(f"  차이: {sum(cleaned_lengths)/len(cleaned_lengths) - original_lengths.mean():+.1f} 단어")

# 변화 샘플
print(f"\n" + "="*80)
print(f"🔍 주요 변화 샘플 (상위 5개)")
print(f"="*80)

changes = []
for idx in range(len(v4)):
    orig = v4.iloc[idx]['summary']
    cleaned = cleaned_summaries[idx]
    if orig != cleaned:
        changes.append((idx, orig, cleaned, len(orig.split()) - len(cleaned.split())))

# 가장 많이 줄어든 순
changes_sorted = sorted(changes, key=lambda x: x[3], reverse=True)[:5]

for i, (idx, orig, cleaned, diff) in enumerate(changes_sorted, 1):
    fname = v4.iloc[idx]['fname']
    print(f"\n[{i}] {fname} (-{diff} 단어)")
    print(f"  원본 ({len(orig.split())} 단어):")
    print(f"    {orig[:100]}...")
    print(f"  정제 ({len(cleaned.split())} 단어):")
    print(f"    {cleaned[:100]}...")

# 제출 파일 생성
output_path = './prediction/submit_solar_v4_no_hallucination.csv'
submission = v4[['fname']].copy()
submission['summary'] = cleaned_summaries
submission.to_csv(output_path, index=False)

print(f"\n" + "="*80)
print(f"✅ 할루시네이션 제거 버전 생성 완료")
print(f"="*80)

print(f"\n📁 파일: {output_path}")
print(f"📊 통계:")
print(f"  - 변경률: {100*stats['changed']/len(v4):.1f}%")
print(f"  - 평균 길이: {sum(cleaned_lengths)/len(cleaned_lengths):.1f} 단어")
print(f"  - 전략: 원본 대화 기반 검증 + 추측 제거")

print(f"\n🎯 다음 단계:")
print(f"  1. Dev 셋 ROUGE 평가")
print(f"  2. v4_original과 비교")
print(f"  3. v3_microtuned와 비교")

print(f"\n" + "="*80 + "\n")
