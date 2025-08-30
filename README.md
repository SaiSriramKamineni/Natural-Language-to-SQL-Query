# Natural Language to SQL Query Agent

## 🎯 Project Overview

This project demonstrates a sophisticated **Natural Language to SQL Query Agent** that can understand human questions in plain English and convert them into SQL queries to extract information from a database. The system then provides natural language answers based on the database results, creating a seamless conversational interface for database interactions.

## 🚀 Key Features

- **Natural Language Understanding**: Converts human questions to SQL queries
- **Multi-Model Support**: Works with both OpenAI GPT-4 and Hugging Face models
- **Intelligent Query Generation**: Automatically generates optimized SQL queries
- **Natural Language Responses**: Converts database results back to human-readable answers
- **Database Schema Awareness**: Understands table structures and relationships
- **Flexible Architecture**: Supports different LLM providers and database types

## 🛠️ Technologies Used

### Core Framework

- **LangChain**: Modern framework for building LLM applications
- **LangChain Core**: Core abstractions and interfaces
- **LangChain Community**: Community-maintained integrations
- **LangChain OpenAI**: OpenAI model integrations
- **LangChain Hugging Face**: Hugging Face model integrations

### Database & Utilities

- **SQLite**: Lightweight, serverless database engine
- **Chinook Database**: Sample music store database for demonstration
- **SQLDatabase**: LangChain utility for database operations

### Environment & Configuration

- **Python-dotenv**: Environment variable management
- **Jupyter Notebooks**: Interactive development and testing

### AI/ML Models

- **OpenAI GPT-4**: Primary language model for query generation
- **Hugging Face Models**: Alternative model support (Qwen2.5-VL-7B-Instruct)

## 🏗️ Architecture & Components

### 1. Database Connection Layer

```python
db = SQLDatabase.from_uri("sqlite:///Chinook.db", sample_rows_in_table_info=0)
```

- Establishes connection to SQLite database
- Configures schema information retrieval
- Provides foundation for all database operations

### 2. Schema Management Functions

```python
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print(f'Query being run: {query} \n\n')
    return db.run(query)
```

- **`get_schema()`**: Retrieves complete database schema information
- **`run_query()`**: Executes SQL queries and returns results
- Enables the system to understand database structure and relationships

### 3. Language Model Management

```python
def get_llm(load_from_hugging_face=False):
    if load_from_hugging_face:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            task="text-generation",
            provider="hyperbolic",
        )
        return ChatHuggingFace(llm=llm)

    return ChatOpenAI(model="gpt-4", temperature=0.0)
```

- **Flexible Model Selection**: Choose between OpenAI and Hugging Face models
- **Temperature Control**: Set to 0.0 for deterministic, consistent outputs
- **Provider Configuration**: Supports different Hugging Face providers

### 4. SQL Query Generation Chain

```python
def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given an input question, convert it to a SQL query. No pre-amble. "
        "Please do not return anything else apart from the SQL query, no prefix aur suffix quotes, no sql keyword, nothing please"),
        ("human", template),
    ])

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
```

- **Prompt Engineering**: Carefully crafted prompts for SQL generation
- **Schema Integration**: Automatically includes database schema in prompts
- **Chain Composition**: Uses LangChain's RunnablePassthrough for data flow
- **Output Parsing**: Ensures clean SQL query extraction

### 5. Natural Language Response Generation

```python
def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""

    prompt_response = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
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
```

- **Multi-Stage Processing**: Combines query generation, execution, and response formatting
- **Chain Orchestration**: Coordinates multiple LangChain components
- **Context Preservation**: Maintains question context throughout the pipeline
- **Natural Language Output**: Converts technical results to human-readable responses

## 🔄 Data Flow Architecture

```
User Question → Schema Retrieval → LLM Query Generation → SQL Execution → LLM Response Generation → Natural Language Answer
     ↓              ↓                    ↓                ↓                ↓                        ↓
  "Give me tracks  Database Schema   SQL Query      Database Results   Formatted Response    "The tracks by the
   by Audioslave"   (Table Info)    (JOIN, WHERE)   (Track Names)     (Natural Language)    artist Audioslave..."
```

## 📊 Database Schema (Chinook)

The project uses the **Chinook Database**, a sample music store database with the following key tables:

- **Artist**: Music artists and bands
- **Album**: Music albums linked to artists
- **Track**: Individual music tracks linked to albums
- **Genre**: Music genres
- **Customer**: Store customers
- **Employee**: Store employees
- **Invoice**: Customer purchase records
- **InvoiceLine**: Individual items in invoices

## 🧪 Example Queries & Use Cases

### 1. Simple Queries

```python
query = "Give me the name of 10 Artists"
# Generates: SELECT Name FROM Artist LIMIT 10
```

### 2. Multi-Column Queries

```python
query = "Give me the name and artist ID of 10 Artists"
# Generates: SELECT Name, ArtistId FROM Artist LIMIT 10
```

### 3. Foreign Key Queries

```python
query = "Give me 10 Albums by the Artist with ID 1"
# Generates: SELECT * FROM Album WHERE ArtistId = 1 LIMIT 10
```

### 4. Table Joins

```python
query = "Give some Albums by the Artist name Audioslave"
# Generates: SELECT Album.* FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'Audioslave'
```

### 5. Complex Multi-Level Joins

```python
query = "Give some Tracks by the Artist name Audioslave"
# Generates: SELECT Track.Name FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'Audioslave'
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Access to OpenAI API (for GPT-4) or Hugging Face models

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ttos
   ```

2. **Install dependencies**

   ```bash
   pip install langchain-core langchain-community langchain-openai langchain-huggingface python-dotenv
   ```

3. **Set up environment variables**

   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook agent.ipynb
   ```

### Configuration

- **OpenAI Setup**: Set your `OPENAI_API_KEY` in the `.env` file
- **Hugging Face Setup**: Configure your preferred provider in the `get_llm()` function
- **Database**: Ensure `Chinook.db` is in the project directory

## 🔧 Development Process

### Phase 1: Foundation Setup

- Database connection establishment
- Basic schema retrieval functions
- Environment configuration

### Phase 2: Core Functionality

- LLM integration (OpenAI GPT-4)
- SQL query generation pipeline
- Basic prompt engineering

### Phase 3: Advanced Features

- Hugging Face model support
- Natural language response generation
- Chain orchestration and data flow

### Phase 4: Testing & Optimization

- Query type validation
- Response quality assessment
- Performance optimization

## 🎨 Design Patterns & Best Practices

### 1. Chain of Responsibility

- Each component has a single responsibility
- Clear separation of concerns
- Easy to modify or extend individual components

### 2. Dependency Injection

- LLM models are injected into functions
- Easy to switch between different providers
- Testable and maintainable code

### 3. Prompt Engineering

- Carefully crafted system and human prompts
- Clear instructions for consistent outputs
- Minimal formatting requirements

### 4. Error Handling

- Graceful fallbacks for different scenarios
- Clear error messages and debugging information
- Robust database operation handling

## 🔍 Technical Challenges & Solutions

### Challenge 1: SQL Query Generation Accuracy

**Problem**: Ensuring generated SQL queries are syntactically correct and semantically accurate
**Solution**:

- Comprehensive schema information in prompts
- Clear system instructions for SQL formatting
- Temperature set to 0.0 for deterministic outputs

### Challenge 2: Multi-Model Support

**Problem**: Supporting different LLM providers with varying APIs
**Solution**:

- Abstracted LLM interface through LangChain
- Provider-agnostic configuration
- Easy switching between models

### Challenge 3: Chain Orchestration

**Problem**: Coordinating multiple processing stages
**Solution**:

- LangChain's RunnablePassthrough for data flow
- Clear input/output contracts between components
- Modular chain composition

## 🚀 Future Enhancements

### 1. Query Validation

- SQL syntax validation before execution
- Query optimization suggestions
- Performance analysis and recommendations

### 2. Enhanced Error Handling

- Better error messages for failed queries
- Query suggestion alternatives
- Fallback mechanisms for complex queries

### 3. Caching & Optimization

- Query result caching
- Schema information caching
- Response generation optimization

### 4. Multi-Database Support

- PostgreSQL, MySQL, and other database types
- Database-specific query optimization
- Connection pooling and management

### 5. User Interface

- Web-based chat interface
- Query history and favorites
- Export functionality for results

## 📚 Learning Outcomes

This project demonstrates several key concepts in modern AI application development:

1. **LangChain Framework**: Understanding of chain composition and orchestration
2. **Prompt Engineering**: Crafting effective prompts for specific tasks
3. **Database Integration**: Seamless connection between AI systems and databases
4. **Multi-Model Support**: Flexibility in choosing different AI providers
5. **Natural Language Processing**: Converting between human language and technical queries
6. **System Architecture**: Designing modular, maintainable AI applications

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:

- Enhanced error handling
- Additional database support
- Query optimization features
- User interface development
- Testing and documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **LangChain Team**: For the excellent framework and documentation
- **OpenAI**: For providing access to GPT-4
- **Hugging Face**: For open-source model hosting
- **Chinook Database**: For the sample database used in development

---

**Note**: This project is designed for educational and demonstration purposes. For production use, consider implementing additional security measures, error handling, and performance optimizations.
#
