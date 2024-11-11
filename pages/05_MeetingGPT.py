import streamlit as st
import math
from pydub import AudioSegment
import glob
import subprocess
import openai
from pydub import AudioSegment
import os 
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore

page_title = "MeetingGPT"
st.set_page_config(
    page_title=page_title,
    page_icon="📹",
)
st.title("📹 "+page_title)

st.markdown('''
            AI가 회의나 인터뷰 영상을 요약하고 질문에 대해 답변해 줍니다.                 
            사이드바에 동영상파일을 업로드 하세요.
            ''')

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
    video = st.file_uploader("Video", type=["mp4","avi","mkv","mov"], disabled=file_load_flag)


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

    
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size = 800,
                chunk_overlap = 50,
                            )
#===============================================================
has_transcript = os.path.exists("./.cache/downloaded_video.txt")

@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript : 
        return
    audio_path = video_path.replace("mp4","mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command,check=True)
    

@st.cache_data()
def cut_audio_inchunks(video_path, chunk_size, chunks_folder):
    if has_transcript : 
        return
    audio_path = video_path.replace("mp4","mp3")
    track = AudioSegment.from_mp3(audio_path)
    
    print("duration : ", track.duration_seconds)
    
    chunk_len = chunk_size*60*1000
    chunks = math.ceil(len(track)/chunk_len)

    for i in range(chunks):
        start_time = i*chunk_len 
        end_time = (i+1)*chunk_len
        chunk = track[start_time: end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3")
        print(f"start : {start_time}, end : {end_time}")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination): 
    
    if has_transcript : 
        return
    
    files =  glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    
    for file in files:
        with open(file, 'rb') as audio_file , open(destination, 'a',encoding="utf-8") as final_file: 
            transcript =openai.Audio.transcribe(
                                    "whisper-1",
                                    audio_file,
                                    language="ko"
                                )
            final_file.write(transcript.get("text", ""))
            


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    '''
    임베딩 및 retriever 수행
    '''    
    # 임배딩결과를 저장할 캐쉬폴더 생성
    cache_dir = LocalFileStore(f"./.cache/meeting_embeddings")
    
    loader = TextLoader(file_path,encoding='utf-8')
    docs = loader.load_and_split(text_splitter=splitter)
    # 임베딩함수 호출
    embeddings = OpenAIEmbeddings()
    # 캐쉬임배딩함수 생성
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # FAISS 벡터스토어사용 >> 임배딩후 >> 파일경로 검색 >> 캐쉬폴더 검사 >> 검색수행 후 캐쉬폴더에 저장 or 캐쉬데이터 사용
    vectorstore = FAISS.from_documents(docs,cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

           
if video : 
    
    video_path = f"./.cache/{video.name}"
    chunks_path = "./.cache/chunks"
    transcript_path = video_path.replace("mp4","txt")
    
    with st.status("Loading video...", expanded=True) as status:
        video_content = video.read()
        
        with open(video_path, "wb") as f : 
            f.write(video_content)
            
        status.update(label = "Extracting audio...")    
        extract_audio_from_video(video_path)
        
        status.update(label = "Cutting audio segment...")
        cut_audio_inchunks(video_path, 10, chunks_path)
    
        status.update(label = "Transcribing audio...")
        transcribe_chunks(chunks_path, transcript_path)
        
    transcript_tab, summary_tab, qa_tab = st.tabs(['transcript_tab','summary_tab','qa_tab'])
    
    with transcript_tab:
        with open(transcript_path, 'r', encoding='utf-8' ) as f:
            st.write(f.read())
            
    with summary_tab:
        start = st.button("Generate summary")
        if start :
            
            llm = ChatOpenAI(
                temperature=0.1,
            )
            
            loader = TextLoader(transcript_path,encoding='utf-8')
            
            docs = loader.load_and_split(text_splitter=splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                                                                    
                """
                )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})
            
            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the oppertunity to refine thte existing summary (only if needed) with some more context below.
                
                --------------
                {context}
                --------------
                Given the new context refine the original summary.
                If the context isn't useful, RETURN the original summary.
                
                """
            )      
            refine_chain = refine_prompt | llm | StrOutputParser()
            with st.status("Summarizing....") as status : 
                for i,doc in enumerate(docs[1:]):
                    status.update(label = f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke({
                        "existing_summary":summary,
                        "context": doc.page_content
                        })
                    st.write(summary)
                    
            st.write(summary)
    
    with qa_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("do they talk about marcus aurelius?")
        st.write(docs)