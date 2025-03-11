I’ve often found that most text-to-SQL agents are quite basic, usually working with simple toy databases. However, in real-world business scenarios, where complexity is much higher, a more advanced approach is necessary to achieve better results.

This project enhances the standard text-to-SQL agent by incorporating two `quadrant` databases:

1- One containing example queries to help the agent refine and complete user inputs.

2- Another focused on improving the WHERE clause, addressing potential issues such as LLM-generated errors, user typos, or formatting inconsistencies (e.g., uppercase vs. lowercase).

Additionally, I’ve implemented safeguards against prompt injection attacks to prevent malicious queries, such as those that could delete tables or expose sensitive data.

To ensure traceability and efficient orchestration, I used `Langsmith` for tracking user inputs/outputs and `LangGraph` to manage the agent’s workflow.

## Setup
1- Make sure to create sqlite db for querying.
2- Run `docker compose up -d` to start the Qdrant vector database, which will be used for Retrieval-Augmented Generation.

## Architecture:
![imagen](https://github.com/user-attachments/assets/02a6e551-7b9a-4cc5-8ab4-ebe4c1324a8c)

---

## Langraph orchestration :
![imagen](https://github.com/user-attachments/assets/19a6e1d2-94de-42d5-98e3-0403a15f55be)

I have relied on the documentation to develop it from:

https://www.uber.com/en-IN/blog/query-gpt/?uclick_id=6cfc9a34-aa3e-4140-9e8e-34e867b80b2b

www.youtube.com/@codingcrashcourses8533
