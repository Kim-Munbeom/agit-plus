import json
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from typing import List
import jq
from datetime import datetime

# JSON 로더 커스텀 함수
def metadata_func(record: dict, metadata: dict) -> dict:
    # 메타데이터에 원본 데이터의 주요 정보 추가
    metadata.update({
        "id": record.get("id", ""),
        "actor_nickname": record.get("actor_nickname", ""),
        "ts": record.get("ts", ""),
    })
    return metadata

# JSON 문서를 로드하고 벡터 DB 생성
@st.cache_resource
def load_and_process_data(file_path: str):
    try:
        # JSONLoader를 사용하여 데이터 로드
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[]',  # 배열의 각 객체를 처리
            content_key="user_message",  # user_message를 주요 컨텐츠로 사용
            metadata_func=metadata_func
        )
        documents = loader.load()

        # 텍스트 분할
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        splits = text_splitter.split_documents(documents)

        # 임베딩 모델 설정 (다국어 모델 사용)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # FAISS 벡터 데이터베이스 생성
        vectordb = FAISS.from_documents(splits, embeddings)

        return vectordb
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.error(f"상세 에러: {str(e)}")
        return None

# Streamlit 앱의 메인 함수
def main():
    st.title("메시지 데이터 검색기")

    # 파일 경로
    main_file_path = "data/data.json"

    # 데이터 로드 및 벡터 DB 생성
    with st.spinner('데이터를 로드하고 있습니다...'):
        vectordb = load_and_process_data(main_file_path)

    if vectordb is None:
        st.error("데이터 로드에 실패했습니다.")
        return

    # 검색 인터페이스
    st.write("### 메시지 검색")
    search_term = st.text_input("검색어를 입력하세요:")

    if search_term:
        with st.spinner('검색 중...'):
            # 유사도 검색 수행
            results = vectordb.similarity_search_with_score(search_term, k=10)

            if results:
                st.write(f"### 검색 결과: {len(results)}개 항목 발견")

                # 결과 표시
                for i, (doc, score) in enumerate(results, 1):
                    # 유사도 점수를 퍼센트로 변환
                    similarity_percentage = (1 - score) * 100

                    # 메시지 작성 시간을 읽기 쉬운 형식으로 변환
                    timestamp = doc.metadata.get('ts', 0)
                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "날짜 없음"

                    with st.expander(f"결과 {i} - 유사도: {similarity_percentage:.2f}% (작성자: {doc.metadata.get('actor_nickname', '알 수 없음')} / 작성일: {date_str})"):
                        # 메시지 내용 표시
                        st.write("#### 메시지 내용")
                        st.write(doc.page_content)

                        # 첨부 파일 정보가 있는 경우 표시
                        if doc.metadata.get('files'):
                            st.write("#### 첨부 파일")
                            for file in doc.metadata['files']:
                                st.write(f"- {file.get('name', '이름 없음')} ({file.get('format', '형식 없음')})")

                        # 이미지가 있는 경우 표시
                        if doc.metadata.get('images'):
                            st.write("#### 첨부 이미지")
                            for img in doc.metadata['images']:
                                if img.get('metadata', {}).get('600_600'):  # 중간 크기 이미지 사용
                                    st.image(img['metadata']['600_600'])

                        # 전체 메타데이터 표시 옵션
                        if st.checkbox(f"상세 정보 보기 #{i}"):
                            st.json(doc.metadata)
            else:
                st.warning("검색 결과가 없습니다.")

    # 데이터 통계 표시
    if vectordb:
        st.sidebar.write("### 데이터 통계")
        st.sidebar.write(f"총 문서 수: {vectordb.index.ntotal}")

if __name__ == "__main__":
    main()
