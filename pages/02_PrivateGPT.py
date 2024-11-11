from typing import Dict, List
from uuid import UUID
import streamlit as st
import time

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory

from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings, OllamaEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama



# 초기화 ---------------------------------------------------------------  

page_title = "PrivateGPT"
st.set_page_config(
    page_title=page_title,
    page_icon="🔒",
)

# 세션초기화     
if "messages" not in st.session_state : 
    st.session_state["messages"] = []
if "memory" not in st.session_state :     
    st.session_state["memory"] = None
    
    
# 함수정의 ---------------------------------------------------------------       

#@st.cahe_data : file이 이전file명과 일치할 경우, 함수를 실행하지 않고 이전 결과값을 반환한다.
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    '''
    사용자 입력파일을 캐쉬폴더에 저장후, 임베딩 및 retriever 수행
    '''
    
    # 입력받은 파일내용 읽기
    file_content = file.read()
    
    # 입력받은 파일을 캐쉬폴더에 저장
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f :
        f.write(file_content)
    
    # 캐쉬폴더에 저장된 파일 로드 
    loader= UnstructuredFileLoader(file_path)
    
    # 스플릿터를 사용하여 파일분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", 
        chunk_size = 600,
        chunk_overlap = 100,
    )
    docs = loader.load_and_split(text_splitter = splitter)
    
    # 임베딩함수 호출
    embeddings = OpenAIEmbeddings()
    
    # 임배딩결과를 저장할 캐쉬폴더 생성
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    # 캐쉬임배딩함수 생성
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # FAISS 벡터스토어사용 >> 임배딩후 >> 파일경로 검색 >> 캐쉬폴더 검사 >> 검색수행 후 캐쉬폴더에 저장 or 캐쉬데이터 사용
    vectorstore = FAISS.from_documents(docs,cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role": role})
    
def send_message(message, role, save=True):
    '''
    메세지를 화면에 출력하고 세션에 저장
    '''
    # 메세지를 화면에 출력
    with st.chat_message(role):
        st.markdown(message)
        
    # 세션에 메세지 누적
    if save :
        save_message(message, role)
        
def paint_history():        
    '''
    히스토리 화면 생성
    '''
    # 세션에 저장된 내역출력 및 세션 중복저장 False
    for message in st.session_state["messages"] : 
        send_message(message["message"],message['role'], save=False)
    
def format_docs(docs):
    '''
    리트리버에 의해 검색된 문서집합을 하나의 문서로 통합
    '''
    return "\n\n".join(document.page_content for document in docs)


    
    
class ChatCallbackHandler(BaseCallbackHandler):
    '''
    callback handler : 
    LangChain의 context 안에 있는 클래스로서,llm의 event를 listen 함.
    답변을 출력할때 스트리밍 효과를 주기위해 생성
    '''    
    message_tk = ""
   
    def on_llm_start(self, *args, **kwargs) : 
         self.message_box = st.empty()
            
    def on_llm_end(self,*args, **kwargs):
        with st.sidebar:
            st.write("llm ended!")
            
        save_message(self.message_tk, "ai")
        
            
    def on_llm_new_token(self, token, *args, **kwargs):
        
        print(token)
        self.message_tk += token
        
        # 전체 메세지에서 요약부분 제거 
        self.message_box.markdown(self.message_tk)


# 위 체인 대신, LCEL을 사용하여 코드를 작성한다. 

def load_memory(_):
    return memory.load_memory_variables({})["history"]

def invoke_chain(question) :
    
    result = chain.invoke(question)
    
    memory.save_context(
        {"input": question},
        {"output": result.content}
        )# 이 부분을 db 에 저장할 수도 있다.
    st.session_state["memory"] = memory
    
    return result
        
# 기본레이어 부 --------------------------------------------------------  

st.markdown("""
            # 🔒PrivateGPT
            로컬환경에서 작동하는 AI챗봇입니다.    
            사이드바에 파일을 업로드 하세요.            
            """)

with st.sidebar : 

    file_load_flag = True
    api_key = st.session_state.get("api_key", None)

    if api_key: 
        if st.session_state.api_key_check :
            st.success("✔️ API confirmed successfully.")  
            file_load_flag = False
            
        else : 
            st.warning("Please enter your API key on the main(home) page.")
    else:
        st.warning("Please enter your API key on the main(home) page.") 
        
    file  = st.file_uploader("Upload a .txt .pdf or .docx file", type = ["pdf", "txt", "docx"], disabled=file_load_flag)
    
    st.components.v1.html(
    """
    <div style = "margin : 50px 0 0 -5px">
     <script type="text/javascript" 
    src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" 
    data-name="bmc-button" 
    data-slug="cj5311" 
    data-color="#FFDD00" 
    data-emoji="" 
    data-font="Cookie" 
    data-text="Buy me a coffee" 
    data-outline-color="#000000" 
    data-font-color="#000000" 
    data-coffee-color="#ffffff"
   >
    </script>
    </div>

    """,
    )

# 사용자 입력 파일이 있을때
if file : 
    llm = ChatOllama(
    model = "mistral:latest",
    # model ="gemma2:2b ",
    temperature = 0.1,
    streaming = True, # 문자 타이핑 플레이 효과, 일부모델에서는 지원안함
    # callbacks = [ChatCallbackHandler()]
    ) 


    # mistral 이 instructor 기반이기 때문에 스트링으로 변환
    prompt = ChatPromptTemplate.from_messages({
        """
        Answer the question using ONLY the following context and not your training data.
        If you don't know the answer just say you don't know.
        Don't make anything up.
        
        Context: {context} 
        Question: {question}
        """
    }
    )


    if st.session_state["memory"] is None:
        
        st.session_state["memory"] = ConversationBufferMemory(
                                        llm = llm,
                                        max_token_limit = 150,
                                        return_messages=True
                                    )


    memory = st.session_state["memory"]
    # rag 수행
    retriever =  embed_file(file)

    # ai 안내문 출력 
    send_message("I'm ready! Ask away!", "ai", save=False)
    
    # 과거 메세지들 화면 출력 
    paint_history()
    
    # 사용자 질문입력창 발생
    message = st.chat_input("Ask anything about your file...")
    
    # 사용자 질문 들어올때
    if message : 
        
        # 사용자 질문을 화면에 출력
        send_message(message, "human")
        
        chain =  {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough(),
            "history" : load_memory
            }  | prompt | llm
        
        with st.chat_message("ai"):
            response = invoke_chain(message)
            
        # st.session_state
        
    else : 
        # 파일을 재업데이트 했을때 세션 초기화
        st.session_state["messages"] = []
        st.session_state["memory"] = None
        
        

    
        
