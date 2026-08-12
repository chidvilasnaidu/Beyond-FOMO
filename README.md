#  Beyond FOMO

### An Agentic AI-Based Information Filtering and Opportunity Assistant

**Beyond FOMO** is an Agentic AI information assistant designed to
reduce information overload while helping users stay connected to
information that actually matters.

## 🚀 Live Demo

👉 **[Live APP Streamlit](https://beyond-fomo-chidvilas.streamlit.app/)**


The project addresses two sides of FOMO:

-   **Social FOMO:** exposure to social comparison, toxic discussions,
    clickbait, repetitive content, and engagement-driven noise.
-   **Information FOMO:** concern that leaving social media may cause
    users to miss jobs, scholarships, visa updates, deadlines,
    government announcements, AI developments, and other opportunities.

> **Core idea:** Don't stay connected to every feed. Stay connected to
> what actually matters.

------------------------------------------------------------------------

##  Features

-   Live news, events, deadlines, and announcement retrieval
-   Evergreen web search for facts, comparisons, rankings, and how-to
    questions
-    Category updates for jobs, visas, scholarships, AI, business, and
    technology
-    FOMO Priority classification: **Critical / Important / Can Wait /
    Ignore**
-    Opportunity detection for scholarships, jobs, internships, visas,
    deadlines, and government announcements
-    Anti-noise moderation for hate speech, gossip, clearly unverified
    claims, source clutter, and promotional noise
-    Preference-based personalization for categories, response style,
    and language
-    Translation support for English, Telugu, Hindi, French, and
    Italian
-    Voice input using Hugging Face Whisper
-   Text-to-speech using Edge TTS
-   Image-to-text using Hugging Face TrOCR
-    LangChain + LangGraph agentic workflow
-    SQLite conversation checkpointing
-    Streamlit multimodal chat interface

------------------------------------------------------------------------

## Architecture

``` text
Text / Voice / Image
        |
        v
Input Processing
        |
        v
Intent Router
   |          |
Feedback    Query
   |          |
   v          v
Feedback   Chat Agent
              |
              v
        Tool Selection
              |
     +--------+---------+---------+
     |        |         |         |
   Search  Category  Weather  AI Tools
     |        |         |         |
     +--------+---------+---------+
              |
              v
       Moderation Layer
              |
              v
         Chat Agent
              |
              v
       Final Response
              |
              v
       Optional Edge TTS
```

The current codebase contains **eight callable specialist tools** and
five graph-level processing nodes. Together they provide more than ten
functional agentic responsibilities, but they are not ten independent
autonomous agents.

### Specialist tools

1.  `SearchEngine`
2.  `WebSearch`
3.  `WeatherTool`
4.  `TranslateTool`
5.  `CategoryNews`
6.  `SummarizeTool`
7.  `FOMOPriority`
8.  `OpportunityFinder`

### Graph nodes

-   `IntentRouter`
-   `ChatAgent`
-   `ToolNode`
-   `ModerateNode`
-   `FeedbackNode`

------------------------------------------------------------------------

## Technology Stack

  Technology                  Purpose
  --------------------------- ----------------------------
  Python                      Core application
  Streamlit                   Frontend
  LangChain                   LLM/tool integration
  LangGraph                   Stateful agentic workflow
  Gemini 2.5 Flash            Main LLM
  Hugging Face Transformers   Whisper and TrOCR
  Whisper Small English       Speech-to-text
  TrOCR Small Printed         Image-to-text
  Edge TTS                    Text-to-speech
  DuckDuckGo Search / DDGS    Web and news retrieval
  SQLite                      Conversation checkpointing
  PyTorch                     Model inference
  NumPy                       Audio processing
  SoundFile                   Audio decoding
  Pillow                      Image processing

------------------------------------------------------------------------

## Prerequisites

-   Python **3.10+** recommended
-   Git
-   Internet connection
-   Google Gemini API key
-   Hugging Face token if required by the model-loading environment

The backend uses `gemini-2.5-flash` and requires `GOOGLE_API_KEY`. The
application also reads `HF_TOKEN` for Hugging Face models.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/chidvilasnaidu/Beyond-FOMO.git
cd Beyond-FOMO
```

Replace the repository URL if your actual GitHub repository name is
different.

### 2. Create a virtual environment

**Windows**

``` bash
python -m venv venv
venv\Scriptsctivate
```

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Create `requirements.txt`:

``` text
streamlit
python-dotenv
langchain-core
langchain
langgraph
langchain-google-genai
langchain-community
ddgs
langgraph-checkpoint-sqlite
transformers
torch
numpy
soundfile
edge-tts
Pillow
sentencepiece
```

Install:

``` bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create `.env` in the project root:

``` env
GOOGLE_API_KEY=your_google_gemini_api_key
HF_TOKEN=your_huggingface_token
```

Never commit `.env` or API keys to GitHub.

------------------------------------------------------------------------

## Run the Application

The project should contain:

``` text
Beyond-FOMO/
├── app.py
├── fomo_agent.py
├── requirements.txt
├── .env
└── README.md
```

Run:

``` bash
streamlit run app.py
```

Then open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## Usage Examples

### Current AI updates

``` text
What are the latest AI developments?
```

### Scholarships

``` text
Find current scholarships for international students.
```

### Visa updates

``` text
What are the latest visa updates?
```

### Jobs

``` text
Show me recent Python or AI job opportunities.
```

### Priority

``` text
Which of these updates are actually important?
```

The FOMO Priority component returns one of:

``` text
Critical
Important
Can Wait
Ignore
```

### Voice

1.  Press the microphone control.
2.  Speak your question.
3.  Review the converted text.
4.  Submit it to FOMO Guardian.

### Image-to-text

1.  Upload a PNG, JPG, JPEG, or WEBP image.
2.  TrOCR extracts printed text.
3.  Review the extracted text.
4.  Submit it to FOMO Guardian.

### Text-to-speech

1.  Ask a question.
2.  Wait for the AI response.
3.  Press **SPEAK**.
4.  Edge TTS generates and plays the response.

------------------------------------------------------------------------

##  End-to-End Workflow

``` text
User Input
   |
   +-- Text
   +-- Voice --> Whisper --> Text
   +-- Image --> TrOCR  --> Text
                     |
                     v
               IntentRouter
                     |
              +------+------+
              |             |
          FEEDBACK        QUERY
              |             |
              v             v
        FeedbackNode    ChatAgent
                            |
                            v
                       Tool Selection
                            |
                            v
                    Retrieval / AI Tools
                            |
                            v
                     Moderation Layer
                            |
                            v
                       ChatAgent
                            |
                            v
                     Final Response
                            |
                            v
                       Edge TTS
```

------------------------------------------------------------------------


## Opportunity Detection

The Opportunity Finder extracts:

-   Scholarships
-   Jobs
-   Internships
-   Visa updates
-   Deadlines
-   Government announcements

The focus is on **actionable information**, not simply increasing news
consumption.

------------------------------------------------------------------------

## Anti-Noise Layer

Retrieved information passes through a moderation stage that attempts
to:

-   Remove hate speech
-   Reduce gossip
-   Remove clearly unverified claims
-   Remove unnecessary source/channel names
-   Remove taglines and "Read more at..." clutter
-   Preserve useful factual information
-   Rewrite results in neutral language

The objective is to **filter noise without removing useful signals**.

------------------------------------------------------------------------

## Multilingual and Audio Processing

Supported translation/output languages:

-   English
-   Telugu
-   Hindi
-   French
-   Italian

The current speech-recognition model is:

``` text
openai/whisper-small.en
```

Therefore, the current ASR implementation is **English-specific**. The
project provides multilingual translation/output capability, but the
current Whisper configuration should not be described as unrestricted
multilingual speech recognition.

### Audio pipeline

``` text
Voice Query
    ↓
Whisper Speech-to-Text
    ↓
LangGraph Agentic Processing
    ↓
AI Response
    ↓
Edge TTS
    ↓
Spoken Response
```

------------------------------------------------------------------------

##  Image-to-Text

The application uses:

``` text
microsoft/trocr-small-printed
```

from Hugging Face.

Supported formats:

``` text
PNG
JPG
JPEG
WEBP
```

Example:

``` text
Image of Notice
      ↓
     TrOCR
      ↓
Extracted Text
      ↓
FOMO Guardian
      ↓
Summary / Explanation / Priority
```

------------------------------------------------------------------------

##  Prompt Engineering

Prompt engineering establishes the behavior of FOMO Guardian. Important
rules include:

-   Use tools for current or time-sensitive information.
-   Distinguish current information from evergreen questions.
-   Produce concise, useful responses.
-   Avoid inventing current facts when tools are unavailable.
-   Apply user preferences.
-   Maintain conversational continuity.
-   Keep source clutter out of final responses.

A separate moderation prompt creates a second quality and
noise-filtering layer after retrieval.

------------------------------------------------------------------------

##  Project Structure

``` text
Beyond-FOMO/
├── app.py                  # Streamlit frontend and multimodal interface
├── fomo_agent.py           # LangGraph backend and specialist tools
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # Local API credentials
├── FOMO_Report_latex.pdf   # Detailed Report
```

------------------------------------------------------------------------

## Security

Add the following to `.gitignore`:

``` gitignore
.env
.streamlit/secrets.toml
__pycache__/
*.pyc
FOMO.db
speech.mp3
venv/
.venv/
```

Never commit Google, Hugging Face, or other API credentials.

------------------------------------------------------------------------

## Limitations

-   The system cannot guarantee that a user will never miss an important
    update.
-   Search coverage, indexing delays, and model errors can affect
    results.
-   Misinformation filtering is probabilistic and does not guarantee
    truth.
-   Priority depends on the user's context.
-   Current Whisper speech recognition is English-specific.
-   The current implementation has eight callable specialist tools, not
    ten independent autonomous agents.
-   The project is an information-management system, not a medical
    diagnosis or treatment system.
-   "Information FOMO" is a project-defined engineering concept, not a
    formally established clinical subtype.

------------------------------------------------------------------------

##  Future Enhancements

-   Source credibility scoring
-   Citation-aware answers
-   Direct source verification
-   Multilingual Whisper / automatic language detection
-   Language-specific TTS voices
-   Scheduled personalized digests
-   Semantic duplicate detection
-   Temporal freshness scoring
-   User-specific priority models
-   Confidence scores
-   Multi-source fact verification
-   Calendar and deadline integration
-   High-priority notification thresholds
-   Signal-to-noise analytics

------------------------------------------------------------------------

##  Contributing

Contributions are welcome.

### Report a bug

1.  Open a GitHub Issue.
2.  Describe the problem.
3.  Include Python version and operating system.
4.  Provide the relevant error message.
5.  Include steps to reproduce the issue.

### Submit a change

``` bash
git checkout -b feature/your-feature
git add .
git commit -m "Add: your feature"
git push origin feature/your-feature
```

Then open a Pull Request with a clear description of the change.

Please do not commit secrets, generated databases, model caches, or
unnecessary files.

------------------------------------------------------------------------

##  License

This project was developed using various open-source materials, libraries, models, resources, and publicly available information from the internet. Some of these resources or information may have been used directly or indirectly during the development of this project.

All rights, ownership, and intellectual property of the original materials remain with their respective authors, creators, organizations, and copyright holders. I do not claim ownership of any third-party materials used in this project.

If you believe that any material has been used inappropriately or without proper attribution, please contact me so that I can review and address the concern.

For any queries or concerns, please contact:
 chidvilasnaidu99@gmail.com
------------------------------------------------------------------------

##  Author

**Kumkapalla Chidvilas**

-   GitHub: https://github.com/chidvilasnaidu
-   LinkedIn: https://www.linkedin.com/in/chidvilas-kumkapalla
-   Hugging Face: https://huggingface.co/chidvilasnaidu

------------------------------------------------------------------------

##  Research Context

Beyond FOMO is motivated by the information-management paradox described
in the project report: social-media consumption can introduce
comparison, negativity, repetition, and engagement-driven noise, while
completely disconnecting can create concern about missing important
information and opportunities.

The project therefore follows a **signal-over-noise** approach:

> **Beyond FOMO does not attempt to keep the user connected to every
> feed. It attempts to keep the user connected to what actually
> matters.**

------------------------------------------------------------------------

##  Acknowledgements

This project uses:

-   LangChain
-   LangGraph
-   Google Gemini
-   Hugging Face Transformers
-   OpenAI Whisper
-   Microsoft TrOCR
-   Edge TTS
-   Streamlit
-   DuckDuckGo Search
