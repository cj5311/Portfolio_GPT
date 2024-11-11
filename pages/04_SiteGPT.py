import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

page_title = "SiteGPT"
st.set_page_config(
    page_title=page_title,
    page_icon="🖥️",
)
st.title("🖥️ "+page_title)

html2txt_transformer = Html2TextTransformer()

st.markdown('''
           AI챗봇이 사이트 내용을 검토하여 답변해 줍니다.   
           사이드바에 사이트 주소를 입력해 주세요.
            ''')

        
api_key = st.session_state.get("api_key", None)
api_key_check = st.session_state.get("api_key_check", None)


if api_key_check : 
    
    llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            api_key=api_key,
        )

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using OnLY the following context answer the user's qewstion.
    If you can't just say you don't know, don't make anything up.
    
    Then, gice a score to the answer between 0 and 5.
    0 being not helpful to the user and 5 being helpful to the user.
    
    Make sure to include the answer's score
    Context : {context}
    
    Examples : 
    Question : How far away is the moon ?
    Answer : The moon is 384,400 km away.
    Score : 5
    
    Question : How far away is the sun?
    Answer : I don't know 
    Score : 0
    
    Your turn !
    
    Question  :{question}
    
    """)

def get_answers(inputs) : 
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm 
    # answers = []
    # for doc in docs : 
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context": doc.apge_content
    #     })
    #     answers.append(result.content)
    # st.write(answers)

    return {"question": question, 
            "answers":[{"answer": answers_chain.invoke({"question": question, 
                           "context":doc.page_content}).content,
             "source":doc.metadata["source"],
             "date": doc.metadata["lastmode"] if "lastmode" in doc.metadata else ""}for doc in docs]}
    
choose_prompt = ChatPromptTemplate.from_messages([
    ("system", '''
     Use ONLY the follwing pre-existing answers to answer the user's question.
     
     Use the answers that have the hightest score (more helpful) and
     favore the most recent ones.
     
     Return the sources of the answers as they are, do not change them.
     
     Answers : {answers}
     
     '''),
    ("human", "{question}")
    ])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm 
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:" "\n" for answer in answers)
    # for answer in answers : 
    #     condensed += f"Answer:{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
    st.write("condensed")
    return choose_chain.invoke({
                "question": question,
                "answers": condensed,
            })
            
# ''''
# 원하는 출력형태  :
# {
#     answer : from the llm, 
#     source : doc.metadata,
#     data : doc
# }
# '''
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000,
    chunk_overlap = 200
)

def parse_page (soup):
    # print(soup)
    header=soup.find("header")
    footer=soup.find("footer")
    
    if header : 
        # text = header.get_text()
        header.decompose()
    if footer : 
        footer.decompose()
                
    str_soup = str(soup.get_text()).replace("\n","").replace("\xa0","").replace("CloseSearch Submit Blog","")
    return str_soup
    

# 함수가 한번 실행한 후 url이 동일하게 인풋됐을때 함수가 수행되지 않고 캐쉬에 저장된 이전값을 반환
@st.cache_data(show_spinner="Loading website..")
def load_website(url):
    loader = SitemapLoader(
        url,
        # filter_urls = [
        #     # "https://openai.com/blog/data-partnerships" # 제외할 url 직접입력
        #     r"^(?!.*\/blog\/).*", #정규표현식을 사용하는 방법
        #     ],
        )
    parsing_function= parse_page
    
    # 1초에 한번만 로딩하는것. 그 이상은 사이트로부터 차단당할 수 있음
    loader.requests_per_second = 1
    # docs = loader.load()
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings()) #캐쉬 사용하기 위해서는 모든URL 마다 각각 저장 필요
    return vector_store.as_retriever()




with st.sidebar:
    file_load_flag = True
    
    if api_key: 
        if st.session_state.api_key_check :
            st.success("✔️ API confirmed successfully.")  
            file_load_flag = False
            
        else : 
            st.warning("Please enter your API key on the main(home) page.")
    else:
        st.warning("Please enter your API key on the main(home) page.") 
        
        
    url = st.text_input("Write down a URL", placeholder="https://excample.com", disabled=file_load_flag )
    
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
    
if url :
    # #async chromium loader : 해당링크의 브라우저를 직접 열어서 사용하기때문에 속도저하로 많은 링크작업에 부적합, 많은양의 자바스크립트를 사용하는 유아이에 작업 가능
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed= html2txt_transformer.transform_documents(docs)
    # st.write(transformed)
    if ".xml" not in url : 
        with st.sidebar : 
            st.error("Please write down a Sitemap URL.")
    else : 
        retriever = load_website(url)
        # docs = retriever.invoke("What is the price of GPT-4")
        query = st.text_input("Ask a question to the website.")
        if query : 
            
            chain = {"docs": retriever,
                    "question": RunnablePassthrough()
                    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            
            result = chain.invoke(query)
            st.markdown(result.content.replace("$","\$"))
