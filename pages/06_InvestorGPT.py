from typing import Any
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool,BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
import streamlit as st
from langchain.schema import SystemMessage
import os

page_title = "InvestorGPT"
st.set_page_config(
    page_title=page_title,
    page_icon="📈",
)

st.title("📈 "+page_title)

st.markdown(
    '''
    AI가 회사정보를 탐색하여 인사이트를 도출해 줍니다.    
    아래에 회사명을 입력하세요.
    '''
    )

with st.sidebar : 
    
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
        
company = st.text_input("Write the name of the company you are interested on.", disabled=file_load_flag)

#======================================================================

        
if company : 
    alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_AIP_KEY")

    llm = ChatOpenAI(
        temperature=0.1,
        model ='gpt-3.5-turbo-1106')

    class StockMarketSymbolSearchToolArgsSchema(BaseModel):
        query : str = Field(description="The query you will search for")

    class StockMarketSymbolSearchTool(BaseTool):
        name = "StockMarketSymbolSearchTool"
        description = '''
        Use this tool to find the stock market symbol for a company.
        It takes a query as an argument.
        Excaple query : Stock Market Symbol for Apple Company.
        '''
        args_schema : Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema
        
        def _run(self,query) :
            ddg = DuckDuckGoSearchAPIWrapper()
            return ddg.run(query)
        
        
    class CompanyOverviewArgsSchema(BaseModel):
        symbol  : str = Field(
            description="Stock symbol of the company. Example : AAPL, TSLA"
        )
        
    class CompanyOverviewTool(BaseTool):
        name = "CompanyOverview"
        description = """
            Use this to get an overview of the financials of the company.
            You should enter a stock symbol.
        """
        args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
        
        def _run(self, symbol) :
            r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
            return r.json()

    class CompanyIncomeStatementTool(BaseTool):
        name = "CompanyIncomeStatement"
        description = """
            Use this to get the income statement of the company.
            You should enter a stock symbol.
        """
        args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
        
        def _run(self, symbol) :
            r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
            return r.json()['annualReports']
        
    class CompanyStockPerformaceStatementTool(BaseTool):
        name = "CompanyStockPerformanceStatement"
        description = """
            Use this to get the weekly performance of a company stock.
            You should enter a stock symbol.
        """
        args_schema : Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
        
        def _run(self, symbol) :
            r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
            response = r.json()
            return list(response["Weekly Time Series"].items())[:200]
        
    
    agent = initialize_agent(
            llm=llm,
            verbose = True,
            agent = AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors =True,

            tools=[
                StockMarketSymbolSearchTool(),
                CompanyOverviewTool(),
                CompanyIncomeStatementTool(),
                CompanyStockPerformaceStatementTool()
            ],
            agent_kwargs={
                "system_message":SystemMessage(content=
                """
                You are a edge fund manager.
                
                You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                
                Consider the performance of a stock, the company overview and the income statement.
                
                Be assertive in your judgement and recommend the stock or advise the user against it.
                
                """)
            }
    )

    prompt = """
    Give me information on Cloudflare's stock, considering its financials, income statements and stock performance help me analyze if it's a potential good investment.
    Also tell me what symbol dose the stock have.
    """

    result = agent.invoke(prompt)
    
    st.write(result['output'])


    