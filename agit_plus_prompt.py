# Python 기본 라이브러리
import streamlit as st
import tiktoken
import json
import boto3
from pathlib import Path
from typing import Any, List, Mapping, Optional, Dict
from datetime import datetime
import os
from pathlib import Path
from comento_func import bedrock_client

# 로깅
from loguru import logger

# Pydantic
from pydantic import BaseModel, Field

# Langchain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.memory import BaseMemory  # BaseMemory만 import
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, BaseMessage)
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.outputs import ChatResult, ChatGeneration

# Langchain
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory  # 여기서 import

# Langchain Community
from langchain_community.document_loaders import (PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def main():
    st.set_page_config(page_title="agi+", page_icon=":bulb:")

    st.title("_Agile Goal Innovation :blue[agi+]_ :bulb:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # main 함수 내에 추가

    if st.session_state.conversation is None:
        data = load_data("data_back")
        if data:
            documents = convert_to_documents(data)
            if documents:
                text_chunks = get_text_chunks(documents)
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
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

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
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
                            logger.info(f"Chain result: {result}")

                            # 응답 처리
                            response = result.get('answer', '')
                            if not response:
                                logger.error("Chain에서 응답을 받지 못했습니다.")
                                response = "응답을 생성할 수 없습니다."
                                st.error(response)
                            else:
                                st.markdown(response)

                    except Exception as e:
                        logger.error(f"Chain 실행 중 오류 발생: {str(e)}", exc_info=True)
                        response = "응답을 생성하는 중 오류가 발생했습니다."
                        st.error(response)

            # 메시지 히스토리 업데이트
            st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


@st.cache_data
def load_data(data_dir: str) -> List[dict]:
    current_dir = Path.cwd()
    logger.info(f"현재 작업 디렉토리: {current_dir}")

    data = []
    data_dir_path = Path(data_dir)
    logger.info(f"찾으려는 데이터 디렉토리 경로: {data_dir_path.absolute()}")

    if not data_dir_path.exists():
        st.error(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir_path}")
        logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir_path}")
        return data

    # users.json 파일 처리
    user_file = data_dir_path / "users.json"
    logger.info(f"users.json 파일 경로: {user_file}")

    if user_file.exists():
        try:
            with open(user_file, 'r', encoding='utf-8') as file:
                users_data = json.load(file)
                st.write(f"users.json에서 {len(users_data)}명의 사용자 정보를 로드했습니다.")
                for user in users_data:
                    processed_item = {
                        "type": "user",
                        "user_message": f"""
                        이름: {user.get('nickname', '')}
                        아지트ID: {user.get('agit_id', '')}
                        소속: {user.get('profile', {}).get('affiliation', '')}
                        이메일: {user.get('email', '')}
                        소개: {user.get('profile', {}).get('introduction', '')}
                        위치: {user.get('profile', {}).get('location', '')}
                        연락처: {user.get('profile', {}).get('phone_number', '')}
                        """.strip(),
                        "raw_data": user
                    }
                    data.append(processed_item)
        except Exception as e:
            st.error(f"users.json 파일 처리 중 오류 발생: {str(e)}")
            logger.error(f"users.json 파일 처리 중 오류: {str(e)}")

    # data.json 파일 처리
    data_file = data_dir_path / "data.json"
    logger.info(f"data.json 파일 경로: {data_file}")

    if data_file.exists():
        try:
            with open(data_file, 'r', encoding='utf-8') as file:
                posts_data = json.load(file)
                processed_count = 0

                def format_timestamp(ts):
                    if ts:
                        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    return ''

                def process_attachments(files, images):
                    attachments = []
                    if files:
                        for file in files:
                            attachments.append(f"""
                            파일: {file.get('name', '')}
                            형식: {file.get('format', '')}
                            크기: {file.get('size', 0)} bytes
                            업로드 시간: {format_timestamp(file.get('created_at'))}
                            """.strip())

                    if images:
                        for image in images:
                            attachments.append(f"""
                            이미지: {image.get('name', '')}
                            형식: {image.get('format_type', '')}
                            크기: {image.get('width', '')}x{image.get('height', '')}
                            업로드 시간: {format_timestamp(image.get('created_at'))}
                            """.strip())

                    return attachments

                def process_post(post):
                    # 기본 메시지 내용 구성
                    message_content = f"""
                    작성자: {post.get('actor_nickname', '익명')}
                    내용: {post.get('user_message', '')}
                    작성시간: {format_timestamp(post.get('ts'))}
                    """

                    # 첨부파일 정보 추가
                    attachments = process_attachments(post.get('files', []), post.get('images', []))
                    if attachments:
                        message_content += "\n첨부파일:\n" + "\n".join(attachments)

                    # 반응 정보 추가
                    reactions = post.get('reactions', {})
                    if reactions:
                        message_content += f"\n반응: 👍 {reactions.get('like_count', 0)} 👎 {reactions.get('dislike_count', 0)}"

                    # 요청 정보 추가
                    request = post.get('request')
                    if request:
                        message_content += f"""
                        \n요청 정보:
                        상태: {request.get('status')}
                        생성시간: {format_timestamp(request.get('created_at'))}
                        """

                    return {
                        "type": "post",
                        "post_content": message_content.strip(),
                        "metadata": {
                            "post_id": post.get('id'),
                            "group_id": post.get('group_id'),
                            "actor_id": post.get('actor_id'),
                            "actor_nickname": post.get('actor_nickname'),
                            "timestamp": post.get('ts'),
                            "is_parent": post.get('is_parent'),
                            "thread_info": {
                                "first_thread_id": post.get('first_thread_id'),
                                "last_thread_id": post.get('last_thread_id'),
                                "threads_count": post.get('threads_count')
                            },
                            "has_files": bool(post.get('files')),
                            "has_images": bool(post.get('images')),
                            "has_reactions": bool(post.get('reactions')),
                            "has_request": bool(post.get('request'))
                        },
                        "raw_data": post
                    }

                # 게시글 데이터 처리
                if isinstance(posts_data, list):
                    for post_group in posts_data:
                        if isinstance(post_group, list):
                            for post in post_group:
                                if isinstance(post, dict):
                                    processed_post = process_post(post)
                                    data.append(processed_post)
                                    processed_count += 1

                logger.info(f"data.json에서 {processed_count}개의 게시글을 처리했습니다.")
                st.write(f"data.json에서 {processed_count}개의 게시글을 로드했습니다.")

        except Exception as e:
            st.error(f"data.json 파일 처리 중 오류 발생: {str(e)}")
            logger.error(f"data.json 파일 처리 중 오류: {str(e)}")

    # 최종 데이터 수 출력
    if data:
        st.write(f"총 {len(data)}개의 데이터를 로드했습니다.")
        logger.info(f"총 {len(data)}개의 데이터 로드 완료")
    else:
        st.warning("로드된 데이터가 없습니다.")
        logger.warning("로드된 데이터가 없습니다.")

    return data


def convert_to_documents(data: List[dict]) -> List[Document]:
    documents = []
    for item in data:
        if item['type'] == 'user':
            doc = Document(
                page_content=item['user_message'],
                metadata={'source': 'user_data', 'type': 'user'}
            )
        else:
            # general 타입의 데이터는 전체 내용을 JSON 문자열로 변환
            doc = Document(
                page_content=json.dumps(item, ensure_ascii=False, indent=2),
                metadata={'source': 'general_data', 'type': 'general'}
            )
        documents.append(doc)
    return documents


def get_text_chunks(data, save_path="text_chunks.json"):
    # 텍스트 청크 파일이 존재하면 로드
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            chunk_contents = json.load(f)
        logger.info(f"텍스트 청크를 {save_path}에서 로드했습니다.")
        return [Document(page_content=content) for content in chunk_contents]

    # 파일이 없으면 새로 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(data)

    # 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump([chunk.page_content for chunk in chunks], f, ensure_ascii=False, indent=2)
    logger.info(f"텍스트 청크가 {save_path}에 저장되었습니다.")
    return chunks


def get_vectorstore(text_chunks, save_path="vectorstore.faiss"):
    save_path = Path(save_path)  # pathlib 사용
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # 벡터 저장소 파일이 존재하면 로드
    if save_path.exists():
        try:
            vectordb = FAISS.load_local(
                str(save_path),
                embeddings=HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask"),
                allow_dangerous_deserialization=True  # 신뢰할 수 있는 파일에서만 True 설정
            )
            logger.info(f"벡터 저장소를 {save_path}에서 로드했습니다.")
            return vectordb
        except Exception as e:
            logger.error(f"벡터 저장소 로드 중 오류 발생: {str(e)}", exc_info=True)
            return None

    # 텍스트 청크가 비어 있는 경우
    if not text_chunks:
        logger.error("텍스트 청크가 비어 있습니다.")
        return None

    try:
        # 임베딩 초기화
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 벡터 저장소 생성
        vectordb = FAISS.from_documents(text_chunks, embeddings)

        # 로컬 저장
        vectordb.save_local(save_path)
        logger.info(f"벡터 저장소가 {save_path}에 저장되었습니다.")

        return vectordb
    except Exception as e:
        logger.error(f"벡터 저장소 생성 중 오류 발생: {str(e)}", exc_info=True)
        return None


def get_conversation_chain(vectorstore):
    if vectorstore is None:
        logger.error("vectorstore가 None입니다.")
        return None

    try:
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

        # 프롬프트 템플릿 정의
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Follow Up Input: {question}

Standalone question:""")

        qa_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.

{context}

Question: {question}

Answer in Korean:""")

        # 체인 생성
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
        )

        return chain

    except Exception as e:
        logger.error("대화 체인 생성 중 오류 발생: %s", str(e))
        return None


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    @classmethod
    def from_llm(
            cls,
            llm: BaseChatModel,
            retriever: BaseRetriever,
            memory: Optional[BaseMemory] = None,
            **kwargs
    ) -> "CustomConversationalRetrievalChain":
        """LLM으로부터 체인을 생성하는 클래스 메서드"""

        # 기본 체인 생성
        chain = super().from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            **kwargs
        )

        # 커스텀 속성 설정
        chain.llm = llm
        return chain

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        """문서를 검색하는 메서드"""
        try:
            docs = self.retriever.invoke(question)
            return self._reduce_tokens_below_limit(docs)
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """체인 실행 메서드"""
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # 관련 문서 검색
        docs = self._get_docs(question, inputs)
        if not docs:
            return {
                "question": question,
                "answer": "문서 검색 중 오류가 발생했습니다.",
                "source_documents": [],
                "chat_history": chat_history
            }

        # 문서 컨텍스트 생성
        context = self._format_documents(docs)

        # 프롬프트 생성
        prompt = self._create_prompt(question, context)

        try:
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that provides accurate information based on the given documents."
                ),
                HumanMessage(content=prompt)
            ]

            response = self.llm.generate([messages])

            if response.generations and response.generations[0]:
                answer = response.generations[0][0].text
            else:
                answer = "응답을 생성할 수 없습니다."

            return {
                "question": question,
                "answer": answer,
                "source_documents": docs,
                "chat_history": chat_history + [
                    HumanMessage(content=question),
                    AIMessage(content=answer)
                ]
            }

        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return {
                "question": question,
                "answer": "응답 생성 중 오류가 발생했습니다.",
                "source_documents": docs,
                "chat_history": chat_history
            }

    def _create_prompt(self, question: str, context: str) -> str:
        return f"""
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
제시할 때에는 반드시 3개 이상의 context를 제시해야 함.
3. 각각의 context 제시할 때에는 반드시 다음의 형식으로 응답해야 함.
[
1. 프로젝트명(혹은 업무주제) :
2. 프로젝트 내용(혹은 업무내용) :
3. agit link : 
]
"""


def _format_documents(self, docs: List[Document]) -> str:
    """문서 포맷팅 메서드"""
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        doc_type = doc.metadata.get('type', 'Unknown')
        formatted_docs.append(
            f"문서 {i} (출처: {source}, 유형: {doc_type}):\n{content}\n"
        )
    return "\n\n".join(formatted_docs)


class ClaudeChatModel(BaseChatModel):
    client: Any = Field(default=None)
    model_name: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.0)

    def __init__(self, bedrock_client, **data):
        super().__init__(**data)
        self.client = bedrock_client

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        formatted_messages = []

        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage)):
                formatted_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                formatted_messages.append({
                    "role": "assistant",
                    "content": message.content
                })

        if not formatted_messages:
            logger.warning("No messages to process")
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="No input provided"),
                generation_info={"finish_reason": "stop"}
            )])

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_messages[-1]["content"]
                        }
                    ]
                }
            ]
        }

        try:
            logger.info(f"Sending request to Claude API with body: {json.dumps(body)}")
            completion = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

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

            logger.info(f"Processed response content: {response_content}")

            # ChatGeneration 객체 생성
            chat_generation = ChatGeneration(
                message=AIMessage(content=response_content),
                generation_info={"finish_reason": "stop"}
            )

            return ChatResult(generations=[chat_generation])

        except Exception as e:
            logger.error(f"Claude API 호출 중 오류 발생: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        return "bedrock_claude"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


if __name__ == '__main__':
    main()