# from crewai import Agent, Crew, Process, Task, LLM
# from crewai.project import CrewBase, agent, crew, task
#
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool, FileWriterTool, FileReadTool
#
# from dotenv import load_dotenv
# load_dotenv()
#
# @CrewBase
# class BlogCrew():
#     """"Blog writing crew"""
#
#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"
#
#     # @agent
#     # def researcher(self) -> Agent:
#     #     return Agent(
#     #         config=self.agents_config['research_agent'], # type: ignore[index]
#     #         tools=[SerperDevTool()],
#     #         verbose=True
#     #     )
#
#     @agent
#     def researcher(self) -> Agent:
#         return Agent(
#             config=self.agents_config['researcher'],
#             llm=LLM(
#                 model="gemini/gemini-2.0-flash",  # Use the gemini/ prefix
#                 api_key=os.getenv("Gemini_API_Key")  # Explicitly pass the key
#             ),
#             verbose=True
#         )
#
#     @agent
#     def writer(self) -> Agent:
#         return Agent(
#             config=self.agents_config['writer_agent'], # type: ignore[index]
#             verbose=True
#         )
#
#     @task
#     def research_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['research_task'], # type: ignore[index]
#             agent = self.researcher()
#         )
#
#     @task
#     def blog_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['blog_task'], # type: ignore[index]
#             agent = self.writer()
#         )
#
#     @crew
#     def crew(self) -> Crew:
#         return Crew(
#             agents=[self.researcher(), self.writer()],
#             tasks=[self.research_task(), self.blog_task()]
#         )
#
# if __name__ == "__main__":
#     blog_crew = BlogCrew()
#     blog_crew.crew().kickoff(inputs={"topic": "The future of electrical vehicles"})

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class BlogCrew():
    """Blog writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        # Initialize a shared LLM instance to keep code clean
        self.gemini_llm = LLM(
            model="gemini/gemini-3.1-pro-preview", # Using stable 2.0
            api_key=os.getenv("Gemini_API_Key"),
            temperature=0.0 # Better for factual research
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            # Key changed from 'researcher' to 'research_agent' to match YAML
            config=self.agents_config['research_agent'],
            llm=self.gemini_llm,
            tools=[SerperDevTool()], # Added tool back in
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'],
            llm=self.gemini_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    @task
    def blog_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_task'],
            agent=self.writer()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # CrewBase automatically collects @agent decorated methods
            tasks=self.tasks,   # CrewBase automatically collects @task decorated methods
            process=Process.sequential,
            verbose=True
        )

if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "The future of electrical vehicles"})