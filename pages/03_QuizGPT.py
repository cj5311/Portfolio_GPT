import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)
    
output_parser = JsonOutputParser()

page_title = "QuizeGPT"
st.set_page_config(
    page_title=page_title,
    page_icon="❓",
)
st.title("❓ " +page_title)

def format_docs(docs):
    '''
    리트리버에 의해 검색된 문서집합을 하나의 문서로 통합
    '''
    return "\n\n".join(document.page_content for document in docs)

        
api_key = st.session_state.get("api_key", None)
api_key_check = st.session_state.get("api_key_check", None)


if api_key_check : 
    llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            api_key=api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
    question_prompt = ChatPromptTemplate.from_messages(
        [("system", """
            
            Youare a helpful assistant that is role playing as a teacher.
            
            Based ONLY on the following context make 10 questions to test the user's Knowledge about the text.
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
            Use (o) to signal the correct answer.
            
            Question examples : 
            
            Question : What is the color of the ocean?
            Answers : Red| Yellow| Green| Blue(o)
            
            Question : what is the capital or Georgia?
            Answers : Baku| Tbilisi(o)| Manila, Beirut
            
            Question : When was Avatar released?
            Answers :  2007| 2001| 2009(o)| 1998
            
            Question : Who was Julius Caesar?
            Answers : A Roman Emperor(o)| Painter| Actor| Model
            
            Your Turn!
            
            Context : {context}
            
            """)
            
            
        ]
        
    )

    question_chains = {"context":format_docs}| question_prompt | llm

    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","""
            You are a powerful formatting algorithm,
            
            You format exam questions into JSON format.
            Answers with (o) are the correct ones.
            
            Example Input:
            
            Question : What is the color of the ocean?
            Answers: Red | Yellow | Green | Blue (o)
            
            Question : what is the capital or Georgia?
            Answers : Baku | Tbilisi(o) | Manila, Beirut
            
            Question : When was Avatar released?
            Answers :  2007 | 2001 | 2009(o)| 1998
            
            Question : Who was Julius Caesar?
            Answers : A Roman Emperor(o) | Painter | Actor | Model
            
            Example Output : 
            
            ```
            json
            {{ "question":[
                    {{
                        "question": "what is the color of the ocean?",
                        "answers":[
                            {{
                                "answer":"Red",
                                "correct":false
                            }},
                            {{
                                "answer":"Yellow",
                                "correct":false
                            }},
                            {{
                                "answer":"Grean",
                                "correct":false
                            }},
                            {{
                                "answer":"Blue",
                                "correct":true
                            }},
                        ]
                    }},
                    
                    {{
                        "question": "what is the capital or Georgia?",
                        "answers":[
                            {{
                                "answer":"Baku",
                                "correct":false
                            }},
                            {{
                                "answer":"Tbilisi",
                                "correct":true
                            }},
                            {{
                                "answer":"Manila",
                                "correct":false
                            }},
                            {{
                                "answer":"Beirut",
                                "correct":True
                            }},
                        ]
                    }},
                    {{
                        "question": "Who was Julius Caesar?",
                        "answers":[
                            {{
                                "answer":"2007",
                                "correct":false
                            }},
                            {{
                                "answer":"2001",
                                "correct":false
                            }},
                            {{
                                "answer":"2009",
                                "correct":true
                            }},
                            {{
                                "answer":"1998",
                                "correct":false
                            }},
                        ]
                    }},
                    {{
                        "question": "When was Avatar released?",
                        "answers":[
                            {{
                                "answer":" A Roman Emperor",
                                "correct":true
                            }},
                            {{
                                "answer":"Painter",
                                "correct":false
                            }},
                            {{
                                "answer":"Actor",
                                "correct":false
                            }},
                            {{
                                "answer":"Model",
                                "correct":false
                            }},
                        ]
                    }}
                    
                ]
            }}
            ```
            
            """)
        ])
        
    formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic):
    # chain.invoke(docs)
    # question_response = question_chains.invoke(docs)
    # st.write(question_response.content)
    # formatting_response = formatting_chain.invoke({
    #     "context": question_response.content
    # })
    # st.write(formatting_response.content)
    
    chain = {"context": question_chains} | formatting_chain | output_parser
    return chain.invoke(docs)

@st.cache_data(show_spinner="Searching Wikipedia....")
def wiki_search(term) : 
    retriver = WikipediaRetriever(
                top_k_results=5, 
                # lang="korea"
                )
    return retriver.get_relevant_documents(term)
    

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    '''
    사용자 입력파일을 캐쉬폴더에 저장후, split
    '''
    # 입력받은 파일내용 읽기
    file_content = file.read()
    cache_file_path = "./.cache/quiz_"
    
    # 입력받은 파일을 캐쉬폴더에 저장
    file_path = cache_file_path+"files/"+file.name
    
    print(file_path)
    
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
    return docs


    
with st.sidebar :
    
    file_load_flag = True
    

    if api_key: 
        if st.session_state.api_key_check :
            st.success("✔️ API confirmed successfully.")  
            file_load_flag = False
            
        else : 
            st.warning("Please enter your API key on the main(home) page.")
    else:
        st.warning("Please enter your API key on the main(home) page.")
        
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.", ("file","Wikipedia Article"), disabled= file_load_flag,
        )
    if choice == "file" : 
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type = ["pdf", "txt", "docx"], disabled= file_load_flag,)
        if file : 
            docs = split_file(file)
    else : 
        topic = st.text_input("Search Wikipedia...")
        
        if topic : 
            
            docs = wiki_search(topic)
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



if not docs : 
    st.markdown("""
                 AI가 출제자가 되어 시험문제를 출력해 줍니다.  
                 사이드파에 파일을 업로드 하거나, 위키피디아 검색을 위한 키워드를 입력하세요.   
                """)
else : 
    
    
    # start = st.button("Generate Quiz")
    # if start: 
    
    response = run_quiz_chain(docs, topic if topic else file.name)
    # st.write(response)
    with st.form("questions_form"):
        for question in response['question']: 
            st.write(question['question'])
            value = st.radio(
                "Select an answer",
                [answers['answer'] for answers in question['answers']],
                                index=None)
            result = {"answer":value,"correct":True} in question['answers']
            if result : 
                st.success('Correct!')
            elif value is not None : 
                st.error('Wrong!')
                
        button = st.form_submit_button()
        
#=================================================