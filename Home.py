import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import time
import streamlit.components.v1 as components

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="😁"
)
        
with st.sidebar:
      
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""  
        
    if "api_key_check" not in st.session_state : 
        st.session_state.api_key_check = False
    
    st.session_state.api_key = st.sidebar.text_input("Enter your API key", type="password")
    
    if (st.session_state.api_key == "") & (st.session_state.api_key_check != False) : 
        st.success("✔️ API confirmed successfully.")
        st.session_state.api_key = st.session_state.api_key_check
        
    else :     
        if st.session_state.api_key:

            try : 
                llm = ChatOpenAI(
                api_key=st.session_state.api_key,
                ) 
                llm.predict("api_test")
                st.success("✔️ API confirmed successfully.")
                st.session_state.api_key_check = st.session_state.api_key
                
            except : 
                st.warning("Invalid API key")  
                # st.session_state.api_key = ""
                st.session_state.api_key_check = False
                
        else:
            st.warning("Please enter your API Key.")   
        

    
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
        
st.markdown(
'''
# GPT-Portfolio.

저의 GPT-Portfolio에 방문해 주셔서 환영합니다!   
GPT를 활용한 서비스 여섯가지를 제공하고 있습니다.   
각 페이지에서 자세한 기능을 살펴보실 수 있습니다.      

- [x] 📃[DocumentGPT](/DocumentGPT)
- [x] 🔒[PrivateGPT](/PrivateGPT)
- [x] ❓[QuizGPT](/QuizGPT)
- [x] 🖥️[SiteGPT](/SiteGPT)
- [x] 📹[MeetingGPT](/MeetingGPT)
- [x] 📈[InvestorGPT](/InvestorGPT)

'''
)

