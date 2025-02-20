import os
import json
from datetime import datetime

def is_date_in_range(filename, start_date, end_date):
    # 파일명이 'YYYY-MM-DD.json' 형식인지 확인하고 날짜 범위 체크
    try:
        # 파일 확장자가 .json인지 확인
        if not filename.endswith('.json'):
            return False

        # 파일명에서 확장자를 제외한 날짜 부분 추출
        date_str = filename[:-5]  # '.json' 제거
        # 날짜 형식으로 파싱
        file_date = datetime.strptime(date_str, '%Y-%m-%d')
        # 날짜 범위 확인
        return start_date <= file_date <= end_date
    except ValueError:
        return False

def merge_json_files_by_date(root_directory, start_date_str, end_date_str):
    try:
        # 입력받은 날짜 문자열을 datetime 객체로 변환
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        if start_date > end_date:
            print("Error: 시작 날짜가 종료 날짜보다 늦을 수 없습니다.")
            return None

    except ValueError:
        print("Error: 날짜 형식이 올바르지 않습니다. 'YYYY-MM-DD' 형식으로 입력해주세요.")
        return None

    # 결과를 저장할 리스트
    merged_data = []
    processed_files = []

    # 모든 하위 디렉토리를 재귀적으로 탐색
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if is_date_in_range(filename, start_date, end_date):
                file_path = os.path.join(root, filename)
                processed_files.append(filename)  # 처리된 파일 기록

                # JSON 파일 읽기
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_data = json.load(file)

                        # 데이터 병합
                        if isinstance(file_data, dict):
                            merged_data.append(file_data)
                        elif isinstance(file_data, list):
                            merged_data.extend(file_data)

                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {str(e)}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

    if not processed_files:
        print(f"지정된 기간({start_date_str} ~ {end_date_str})에 해당하는 JSON 파일을 찾을 수 없습니다!")
        return None

    # 처리된 파일들을 날짜순으로 정렬하여 출력
    processed_files.sort()
    print("\nProcessed files:")
    for file in processed_files:
        print(f"- {file}")

    # 최종 데이터를 배열로 한번 더 감싸기
    final_data = [merged_data]

    # 병합된 데이터를 새 파일에 저장
    output_filename = f'merged_{start_date_str}_to_{end_date_str}.json'
    output_path = os.path.join(root_directory, output_filename)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(final_data, outfile, ensure_ascii=False, indent=4)

    print(f"\nMerged file created at: {output_path}")
    print(f"Total files merged: {len(processed_files)}")

    return output_path

def main():
    # 사용자로부터 날짜 입력 받기
    print("날짜는 'YYYY-MM-DD' 형식으로 입력해주세요.")
    start_date = input("시작 날짜를 입력하세요: ")
    end_date = input("종료 날짜를 입력하세요: ")

    directory = './data'  # data 폴더 경로
    merged_file = merge_json_files_by_date(directory, start_date, end_date)

if __name__ == "__main__":
    main()
