from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub

#from Tools import

from langchain_community.llms import Ollama
from my_tools import CustomTool


class llama_agent():
    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.instructions = """You're a professional mine data analyst. 
        Your responsibility is to analyze the mine situation for the user based on the existing data, 
        including but not limited to various data trends, safety risks, construction recommendations, etc. You must give me your thought."""
        self.base_prompt = hub.pull("langchain-ai/react-agent-template")
        self.prompt = self.base_prompt.partial(instructions=self.instructions)
        self.tools = None

    def set_tools(self, tools_list):
        self.tools = tools_list

    def create_agent(self):
        self.my_agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agentexecutor = AgentExecutor(agent=self.my_agent, tools=self.tools, verbose=True, handle_parsing_errors=True)


    # 这是拿invoke定义一个简单的chat函数，只给出final answer
    def chat(self, message):
        response = self.agentexecutor.invoke({"input": str(message)})
        pass
        return response["output"]

if __name__ == "__main__":
    test=llama_agent()
    test.set_tools([CustomTool()])
    test.create_agent()
    response = test.chat("Introduce yourself.")

    print(response)
