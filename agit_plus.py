import streamlit as st
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Any, List, Mapping, Optional, Dict
from pydantic import Field
import boto3
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import (AIMessage, BaseMessage, ChatGeneration, ChatResult, HumanMessage, SystemMessage)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    # .env 파일 로드
    load_dotenv()

    st.set_page_config(page_title="agi+", page_icon=":bulb:")

    st.title("_Agile Goal Innovation :blue[agi+]_ :bulb:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if st.session_state.conversation is None:
        # RAG 준비
        if Path("./vector-store").joinpath("index.faiss").exists():
            vector = vector_store(None)
            if vector:
                st.session_state.conversation = chin_conversation(vector)
                if st.session_state.conversation is None:
                    st.error("대화 체인을 생성할 수 없습니다.")
        else:
            json_data = get_json_file()
            if json_data:
                document = convert_data(json_data)
                if document:
                    chunk = text_chunk(document)
                    if chunk:
                        vector = vector_store(chunk)
                        if vector:
                            st.session_state.conversation = chin_conversation(vector)
                            if st.session_state.conversation is None:
                                st.error("대화 체인을 생성할 수 없습니다.")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {
                "role": "assistant",
                "content": "안녕하세요! Agi+에 오신 것을 환영합니다.\n협업툴 Agit에 저장된 데이터와 기록을 바탕으로, 과거 히스토리를 분석하여 문제 해결을 도와드립니다. 아래와 같은 순서로 대화를 진행하려 합니다. 😊 \n1. 먼저, 해결하고자 하는 문제나 찾고자 하는 히스토리를 알려주세요.\n2. 입력하신 내용을 기반으로 관련 기록을 검색합니다.\n3. 검색 결과를 요약하고, 필요 시 추가 정보 탐색을 진행합니다.\n4. 최종적으로 답변을 확정하여 문제 해결을 돕습니다. 어떤 문제를 도와드릴까요?"
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is None:
                logger.error("대화 체인이 초기화되지 않았습니다.")
                response = "시스템이 준비되지 않았습니다. 페이지를 새로고침 해주세요."
                st.error(response)
            else:
                with st.spinner("업무 히스토리를 찾는 중이에요..."):
                    try:
                        # 입력 검증
                        if not query.strip():
                            response = "질문을 입력해주세요."
                        else:
                            # Chain 실행
                            result = chain({"question": query})
                            # logger.info(result["source_documents"])

                            # 응답 처리
                            response = result.get('answer', '')
                            source_document = result.get('source_documents', '')
                            if not response:
                                logger.error("Chain에서 응답을 받지 못했습니다.")
                                response = "응답을 생성할 수 없습니다."
                                st.error(response)
                            else:
                                st.subheader("답변")
                                st.markdown(response)
                                # st.subheader("참고 자료")
                                # for doc in source_document:
                                #     st.write(doc.page_content)
                                #     st.write(doc.metadata["url"])
                                #     st.write("=================")

                    except Exception as e:
                        logger.error(f"Chain 실행 중 오류 발생: {str(e)}", exc_info=True)
                        response = "응답을 생성하는 중 오류가 발생했습니다."
                        st.error(response)

            # 메시지 히스토리 업데이트
            st.session_state.messages.append({"role": "assistant", "content": response})


# json 파일
@st.cache_data
def get_json_file():
    root_dir = Path("./data")  # 탐색할 루트 디렉토리
    data = []
    count = 0
    # 루트 디렉토리 안의 모든 폴더 탐색
    for file_path in root_dir.rglob("*"):
        logger.info(file_path.name)
        for json_file in file_path.rglob("*"):
            try:
                # json 파일 오픈
                with open(json_file, "r", encoding="utf-8") as file:
                    # json 파일 로드
                    loader = json.load(file)
                    for item in loader:
                        data.append(item)
                        count += 1
            except FileNotFoundError:
                print("파일을 찾을 수 없습니다.")
            except json.JSONDecodeError:
                print("JSON 형식이 올바르지 않습니다.")
    logger.info(f"json 파일에서 {count}개의 데이터를 처리했습니다.")
    return data


# 데이터 변환하기(딕셔너리 리스트 -> 객체 리스트)
def convert_data(data):
    document = []
    for item in data:
        try:
            ts = datetime.fromtimestamp(item['ts']).strftime('%Y-%m-%d %H:%M:%S')
            agit_url = f"https://comento.agit.io/g/{item['group_id']}/wall/{item['id']}" if item[
                'is_parent'] else f"https://comento.agit.io/g/{item['group_id']}/wall/{item['first_thread_id']}#comment_panel_{item['id']}"
            convert = Document(
                # 텍스트 분할이 적용되는 데이터
                page_content=item['user_message'],
                # 분할된 데이터의 정보
                metadata={
                    'id': item['id'],
                    'parent_id': item['first_thread_id'],
                    'writer': item['actor_nickname'],
                    'type': '글' if item['is_parent'] else '댓글',
                    'url': agit_url,
                    'ts': ts
                }
            )
            document.append(convert)
        except Exception as e:
            logger.error(f"데이터 변환하기(딕셔너리 리스트 -> 객체 리스트) 중 오류 {str(e)}")
    return document


# 텍스트 분할(청크)
def text_chunk(data, chunk_save_path=Path("chunk-store"), is_save=False):
    try:
        if is_save is False and chunk_save_path.joinpath("text_chunks.json").exists():
            with open(chunk_save_path.joinpath("text_chunks.json"), "r", encoding="utf-8") as file:
                chunks = json.load(file)
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
                separators=[
                    "\n\n",  # 단락 구분
                    "\n",  # 줄바꿈
                    ".",  # 온점
                    "!",  # 느낌표
                    "?",  # 물음표
                    ",",  # 쉼표
                    " ",  # 공백
                    ""  # 문자 단위
                ]
            )
            chunks = text_splitter.split_documents(data)
            # 저장
            with open(chunk_save_path.joinpath("text_chunks.json"), 'w', encoding='utf-8') as file:
                json.dump([chunk.page_content for chunk in chunks], file, ensure_ascii=False, indent=2)
            logger.info(f"텍스트 청크가 {chunk_save_path}에 저장되었습니다.")

        return chunks
    except Exception as e:
        logger.error(f"텍스트 분할(청크) 중 오류 {str(e)}")


# 임베딩 및 벡터 저장소
def vector_store(data=None, vector_save_path=Path("./vector-store"), is_save=False):
    try:
        # 벡터 저장소 로컬 파일 확인
        if is_save is False and vector_save_path.joinpath("index.faiss").exists():
            vector = FAISS.load_local(
                str(vector_save_path),
                embeddings=HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
                allow_dangerous_deserialization=True  # 신뢰할 수 있는 파일에서만 True 설정
            )
        else:
            # 임베딩
            embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # 벡터 저장소 생성
            vector = FAISS.from_documents(data, embeddings)
            # 벡터 저장소 로컬 저장
            vector.save_local(str(vector_save_path))

            logger.info("임베딩 및 벡터 저장소 생성/저장 완료")
        return vector
    except Exception as e:
        logger.error(f"임베딩 및 벡터 저장소 생성/저장 중 오류 {str(e)}")


def chin_conversation(data):
    bedrock_client = boto3.client(
        service_name=os.getenv("SERVICE_NAME"),
        region_name=os.getenv("REGION_NAME"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # LLM 모델
    llm = ClaudeChatModel(
        bedrock_client=bedrock_client,
        model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=2048,
        temperature=0.0
    )

    # 메모리 초기화 수정
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
        input_key='question'
    )

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    이전 대화 내용: {chat_history}
현재 질문: {question}

위 대화 흐름을 고려하여 현재 질문을 독립적이고 완성된 질문으로 만들어주세요.
변환된 질문:
    """)

    question_prompt = PromptTemplate.from_template("""
    question
1. 정의 : 사용자가 입력한 프롬프트, 현재의 업무를 해결 혹은 진행하기 위한 목적하에서 과거의 업무 히스토리를 참고하고자 하는 내용
2. 변수 : {question}

context
1. 정의 : 사용자가 해결, 확인하고자 하는 업무상의 히스토리를 학습한 데이터에서 높은 유사도를 가진 내용으로 불러온 것
2. 변수 : {context}

역할
1. 너는 사용자가 입력한 question을 기반으로 제공된 context를 사용자에게 전달하는 챗봇임.
2. 이 때, 제공된 context가 사용자에게 유용하도록 question의 내용과 유사도가 높아야 함.

답변
1. 사용자가 입력한 question을 확인하고, 키워드 분석 및 문제 상황을 정의
2. 분석된 키워드 및 문제 상황에 기반해 유사도 높은 context를 탐색하고, 사용자에게 이를 제시
    """)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=data.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": question_prompt},
        return_source_documents=True,
        verbose=True
    )

    return chain


class ClaudeChatModel(BaseChatModel):
    """
    Amazon Bedrock의 Claude 모델을 사용하기 위한 채팅 모델 클래스
    BaseChatModel을 상속받아 구현
    """

    # 클래스 속성 정의
    client: Any = Field(default=None)  # Bedrock 클라이언트 객체
    model_name: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")  # 사용할 Claude 모델 버전
    max_tokens: int = Field(default=2048)  # 생성할 최대 토큰 수
    temperature: float = Field(default=0.0)  # 생성 텍스트의 무작위성 정도 (0.0: 결정적, 1.0: 매우 창의적)

    def __init__(self, bedrock_client, **data):
        """
        초기화 함수
        Args:
            bedrock_client: AWS Bedrock 클라이언트 객체
            **data: 추가 설정 매개변수
        """
        super().__init__(**data)
        self.client = bedrock_client

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> ChatResult:
        """
        메시지를 받아 Claude API를 호출하여 응답을 생성하는 핵심 메서드

        Args:
            messages: 입력 메시지 리스트
            stop: 생성을 중단할 토큰 리스트 (옵션)
            run_manager: 콜백 관리자 (옵션)
            **kwargs: 추가 키워드 인자

        Returns:
            ChatResult: 생성된 응답을 포함하는 결과 객체
        """

        # 입력 메시지가 없는 경우 처리
        if not messages:
            logger.warning("No messages to process")
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="No input provided"),
                generation_info={"finish_reason": "stop"}
            )])

        # 마지막 메시지만 추출
        last_message = messages[-1]

        # 메시지 타입에 따른 내용 추출
        if isinstance(last_message, (HumanMessage, SystemMessage, AIMessage)):
            message_content = last_message.content
        else:
            logger.warning(f"Unsupported message type: {type(last_message)}")
            message_content = str(last_message.content)

        # Claude API 요청 본문 구성
        body = {
            "anthropic_version": "bedrock-2023-05-31",  # Claude API 버전
            "max_tokens": self.max_tokens,  # 최대 토큰 수
            "temperature": self.temperature,  # 온도 설정
            "messages": [
                {
                    "role": "user",  # 사용자 역할로 메시지 전송
                    "content": [
                        {
                            "type": "text",
                            "text": message_content
                        }
                    ]
                }
            ]
        }

        try:
            # API 요청 로깅
            logger.info(f"Sending request to Claude API with body: {json.dumps(body)}")

            # Bedrock API 호출
            completion = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            # 응답 파싱
            response = json.loads(completion["body"].read())
            logger.info(f"Received response from Claude API: {json.dumps(response)}")

            # 응답 구조 확인 및 처리
            if "content" in response and len(response["content"]) > 0:
                response_content = response["content"][0]["text"]
            elif "completion" in response:
                response_content = response["completion"]
            else:
                logger.error(f"Unexpected response structure: {json.dumps(response)}")
                response_content = "응답을 처리하는 중 오류가 발생했습니다."

            # 처리된 응답 내용 로깅
            logger.info(f"Processed response content: {response_content}")

            # ChatGeneration 객체 생성 및 반환
            chat_generation = ChatGeneration(
                message=AIMessage(content=response_content),
                generation_info={"finish_reason": "stop"}
            )

            return ChatResult(generations=[chat_generation])

        except Exception as e:
            # 오류 발생 시 로깅 및 예외 전파
            logger.error(f"Claude API 호출 중 오류 발생: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        """
        LLM 타입을 반환하는 프로퍼티
        Returns:
            str: LLM 타입 식별자
        """
        return "bedrock_claude"

    @property
    def _identifying_params(self) -> dict:
        """
        모델의 식별 파라미터를 반환하는 프로퍼티
        Returns:
            dict: 모델 식별을 위한 파라미터 딕셔너리
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


if __name__ == '__main__':
    main()
