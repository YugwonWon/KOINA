#!/usr/bin/env python3
"""
JSON 파일들을 파싱하여 TSV 파일을 생성하는 스크립트
133.감성 및 발화 스타일 동시 고려 음성합성 데이터의 JSON 파일들을 처리합니다.

출력 TSV 칼럼: filename, sex, text
"""

import json
import os
import glob
from pathlib import Path
import argparse


def parse_json_file(json_path: str) -> list:
    """
    단일 JSON 파일을 파싱하여 (filename, sex, text) 튜플 리스트 반환
    """
    results = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON이 리스트인 경우 (여러 데이터가 배열로 묶여있음)
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        
        for item in items:
            # 성별 정보 추출
            gender = item.get('reciter', {}).get('gender', 'UNKNOWN')
            # MALE -> M, FEMALE -> F로 변환
            if gender == 'MALE':
                sex = 'M'
            elif gender == 'FEMALE':
                sex = 'F'
            else:
                sex = 'U'  # Unknown
            
            # sentences 처리
            sentences = item.get('sentences', [])
            for sentence in sentences:
                sentence_id = sentence.get('id', '')
                
                # voice_piece에서 filename 또는 id를 사용
                voice_piece = sentence.get('voice_piece', {})
                filename = voice_piece.get('filename', '')
                
                # filename이 없으면 id + .wav 사용
                if not filename:
                    filename = f"{sentence_id}.wav"
                
                # 텍스트는 origin_text 사용
                text = sentence.get('origin_text', '')
                
                if filename and text:
                    # .wav 확장자 제거
                    filename_without_ext = filename.replace('.wav', '')
                    results.append((filename_without_ext, sex, text))
    
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {json_path} - {e}")
    except Exception as e:
        print(f"파일 처리 오류: {json_path} - {e}")
    
    return results


def process_directory(input_dir: str, output_file: str):
    """
    디렉토리 내 모든 JSON 파일을 처리하여 TSV 파일 생성
    """
    # JSON 파일 검색
    json_files = glob.glob(os.path.join(input_dir, '**', '*.json'), recursive=True)
    
    print(f"발견된 JSON 파일 수: {len(json_files)}")
    
    all_results = []
    processed = 0
    
    for json_path in json_files:
        results = parse_json_file(json_path)
        all_results.extend(results)
        processed += 1
        
        if processed % 100 == 0:
            print(f"처리 중... {processed}/{len(json_files)} 파일 완료")
    
    print(f"총 {len(all_results)}개의 문장 추출 완료")
    
    # TSV 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("filename\tsex\ttext\n")
        
        for filename, sex, text in all_results:
            # 텍스트 내 탭, 줄바꿈 제거
            text_cleaned = text.replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            f.write(f"{filename}\t{sex}\t{text_cleaned}\n")
    
    print(f"TSV 파일 저장 완료: {output_file}")
    
    # 통계 출력
    male_count = sum(1 for _, sex, _ in all_results if sex == 'M')
    female_count = sum(1 for _, sex, _ in all_results if sex == 'F')
    unknown_count = sum(1 for _, sex, _ in all_results if sex == 'U')
    
    print(f"\n=== 통계 ===")
    print(f"남성(M): {male_count}")
    print(f"여성(F): {female_count}")
    print(f"미상(U): {unknown_count}")
    print(f"총합: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description='JSON 파일들을 파싱하여 TSV 파일 생성')
    parser.add_argument('--input', '-i', type=str, 
                        default='data/style',
                        help='입력 디렉토리 경로')
    parser.add_argument('--output', '-o', type=str,
                        default='data/133_parsed_output.tsv',
                        help='출력 TSV 파일 경로')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 존재 확인
    if not os.path.exists(args.input):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {args.input}")
        return
    
    process_directory(args.input, args.output)


if __name__ == '__main__':
    main()
