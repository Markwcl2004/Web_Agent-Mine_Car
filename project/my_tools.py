from langchain.tools import BaseTool
import numpy as np

class CustomTool(BaseTool):
    name = "Temperature Detector"
    description = "useless tool"
 
    def _run(self, input: str) -> str:
        print("Tools 1 works well and free!")
        # Your logic here
        return "temperature is not bad,huh,20 celceius"
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

class Warning(BaseTool):
    name = "Warn Everyone To Leave"
    description = "When the 'mine condition' is examined to be 'dangerous'. Use this tools to warn everyone to leave out the mine."
 
    def _run(self, input: str) -> str:
        print("Tools 2 works well and free!")
        # Your logic here
        return "Raising up arm. "
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

class Check_condition(BaseTool):
    name = "Check the Condition of Mine"
    description = "This is a custom tool for you to check the condition of the mine, including temperature, huminity, density of some harmful gases and so on."
 
    def _run(self, input: str) -> str:
        print("Tools 3 works well and free!")
        # Your logic here
        return "Putting down arm "
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class Generate_summarize(BaseTool):
    name = "Open Temperature Detector"
    description = "This is a custom tool for open temperature detector"
 
    def _run(self, input: str) -> str:
        print("Tools 4 works well and free!")
        # Your logic here
        return "temperature detector working "
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
 
    
