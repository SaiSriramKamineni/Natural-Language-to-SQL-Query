🧠 Natural Language to SQL Query Agent
🎯 Project Overview

This project demonstrates a sophisticated Natural Language to SQL Query Agent capable of understanding plain English questions and translating them into SQL queries to retrieve relevant data from a database. It also converts query results into human-readable responses—providing a seamless, conversational interface for database interaction.

🚀 Key Features

Natural Language Understanding: Converts user questions into SQL queries

Multi-Model Support: Supports OpenAI GPT-4 and Hugging Face models

Intelligent Query Generation: Generates optimized SQL statements

Human-Readable Responses: Translates SQL results into natural language

Schema Awareness: Understands database tables and relationships

Modular Architecture: Supports different LLM providers and database engines

🛠️ Technologies Used
Frameworks

LangChain: Framework for building LLM-powered applications

LangChain Core/Community/OpenAI/HuggingFace: Model and integration modules

Databases

SQLite: Lightweight database engine

Chinook Database: Sample music store database

SQLDatabase (LangChain): Utility for DB operations

Utilities

Python-dotenv: Environment variable management

Jupyter Notebooks: For development and testing

Models

OpenAI GPT-4: Default LLM for SQL generation and response

Hugging Face (Qwen2.5-VL-7B-Instruct): Alternative open-source LLM

🏗️ Architecture & Components
1. Database Connection Layer
db = SQLDatabase.from_uri("sqlite:///Chinook.db", sample_rows_in_table_info=0)


Connects to SQLite database

Retrieves schema details

Enables query execution

2. Schema Management
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print(f'Query being run: {query}\n')
    return db.run(query)


get_schema(): Fetches database schema

run_query(): Executes a given SQL query

3. Language Model Selection
def get_llm(load_from_hugging_face=False):
    if load_from_hugging_face:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            task="text-generation",
            provider="hyperbolic",
        )
        return ChatHuggingFace(llm=llm)

    return ChatOpenAI(model="gpt-4", temperature=0.0)


Choose between OpenAI or Hugging Face

Temperature set to 0.0 for deterministic outputs

4. SQL Query Generation Chain
def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given an input question, convert it to a SQL query. No preamble. "
        "Return only the SQL query—no quotes, no comments, no extra text."),
        ("human", template),
    ])

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


Structured prompting ensures correct and clean SQL generation

Schema-aware LLM prompting for accuracy

5. Natural Language Answer Generation
def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt_response = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a question and SQL response, convert it to a natural language answer. No preamble.",
        ),
        ("human", template),
    ])

    full_chain = (
        RunnablePassthrough.assign(query=write_sql_query(llm))
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | llm
    )

    return full_chain.invoke({"question": query})


Combines all stages: question → SQL → result → answer

Provides final human-readable response

🔄 Data Flow
User Input → Schema Retrieval → SQL Query Generation → SQL Execution → Response Generation → Final Answer
    ↓              ↓                   ↓                     ↓                   ↓                  ↓
"Tracks by   → [Track, Artist] → SELECT ... JOIN ... → [Results] → "The tracks by Audioslave are..."
 Audioslave"

📊 Chinook Database Overview

Tables used:

Artist: Music artists and bands

Album: Albums linked to artists

Track: Tracks linked to albums

Genre: Music genres

Customer: Music store customers

Employee: Store staff

Invoice: Purchase records

InvoiceLine: Items in each invoice

🧪 Example Use Cases
1. Basic Query
query = "Give me the name of 10 Artists"
# → SELECT Name FROM Artist LIMIT 10

2. Multi-Field Query
query = "Give me the name and artist ID of 10 Artists"
# → SELECT Name, ArtistId FROM Artist LIMIT 10

3. Filter by Foreign Key
query = "Give me 10 Albums by the Artist with ID 1"
# → SELECT * FROM Album WHERE ArtistId = 1 LIMIT 10

4. Table Join
query = "Give some Albums by the Artist name Audioslave"
# → SELECT Album.* FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'Audioslave'

5. Multi-Level Join
query = "Give some Tracks by the Artist name Audioslave"
# → SELECT Track.Name FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'Audioslave'

🚀 Getting Started
Requirements

Python 3.8+

Jupyter Notebook/Lab

OpenAI or Hugging Face API access

Installation
git clone <repository-url>
cd ttos
pip install langchain-core langchain-community langchain-openai langchain-huggingface python-dotenv

Configuration
# .env file setup
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

Run the Notebook
jupyter notebook agent.ipynb

🔧 Development Process
Phase 1: Setup

DB connection

Schema inspection

Env config

Phase 2: Core Features

GPT-4 SQL generation

Prompting logic

Query execution

Phase 3: Expansion

Hugging Face model support

Response generation

LangChain orchestration

Phase 4: Testing

Query accuracy tests

Performance tuning

Error scenarios

🎨 Best Practices & Patterns

Chain of Responsibility: Each component has a single job

Dependency Injection: Easily switch models

Prompt Engineering: Clear, deterministic outputs

Modular Design: Replaceable parts and testable functions

🔍 Technical Challenges & Solutions
1. SQL Accuracy

Issue: Incorrect SQL generation

Fix: Rich schema prompts, deterministic output, strict formatting

2. Model Support

Issue: Multiple LLM APIs

Fix: Abstract model interfaces

3. Chain Flow

Issue: Complex chaining

Fix: LangChain RunnablePassthrough for clean data flow

🔮 Future Improvements

✅ SQL query validation and optimization

✅ Error messaging and fallbacks

✅ Query caching and schema caching

✅ Multi-DB support (PostgreSQL, MySQL)

✅ Web-based UI with query history

📚 Learning Outcomes

LangChain chain orchestration

Prompt engineering for LLMs

SQL query generation from NL

Schema-based AI reasoning

OpenAI/HF integration

Modular software architecture

🤝 Contributing

You're welcome to contribute!

Focus areas:

More DB engines

Improved prompts

Caching strategies

UI development

Testing and coverage

📄 License

This project is open source under the MIT License
.

🙏 Acknowledgments

LangChain – For their open-source framework

OpenAI – For GPT-4 API access

Hugging Face – For open-source models

Chinook Database – For a great SQL demo dataset

