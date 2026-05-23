# Ollama Mastery

## What are llms:
    LLMs are nothing but the collections of the wieghts and the biases. They are complex neural network and they have many layers and connections. when we train a LLM everything it learns are basically stored in the terms of wieghts and biases.

## Proprietary Models and Open Source Models

### Proprietary Models:
    These are the models mostly owned and controlled bu the big company/organizations.
* They are often refered to `black boxes` as their source code, training data andn learned parameters are not publically accessible, preventing external users from fully inspecting or understanding the internal decision maning process.
* These models are typically accessed through paod subscriptions or usage-based pricing although limited free access may be provided for testing or evaluation purposes.
* Properietary models are commonly deployed as cloud-based services where users send requests to the providers infrastructure and recieve responses without direct access to the underlying model. 

* Examples: 
    GPT models from OpenAI, Gemini from Google

### Open Source Model:
    These are the AI models whose core details are openly shared with the public:
* There model architecture, weights and sometimes even the training data are made public. Anyone can download them, study how they work, modify them, fine-tune them for their own needs, run them on their own systems.
* you can download the model files(weights) from platforms like `Hugging Face` and run them for free.
* They are customizable as you have the files, you can `fine-tune` the model on your own private data to make it an expert in a specific field.

* Example:
    Meta's Llama 3, Mistral, Deepseek


## There are free Open Source models available but people still choose to go for the Proprietary Models?
    While open-source models offer transparency and customization, users continue to pay for proprietary models primarily due to liability protection, infrastructure convenience, and enterprise-grade support.  The primary frictions of using open-source models locally involve significant hidden operational costs, complexity in deployment, and lack of legal indemnification.

### Operational and financial Frictions:
* `The "Free" Illusion`: Open-source models shift costs from licensing fees to internal budgets, requiring expensive specialized headcount for MLOps, model security, and performance monitoring. 
* `Infrastructure Burden`: Running enterprise-grade models locally demands substantial, persistent GPU clusters and cooling infrastructure, which carries heavy financial and operational weight.
* `Incident Recovery`: Unlike proprietary APIs, if a self-hosted model fails or produces errors, the organization bears the full cost of diagnosis and recovery without vendor assistance. 

### Technical and Governance Frictions:
* `Liability and Risk`: Deploying open-source models means the organization inherits 100% of the liability for outputs, including copyright infringement or defamatory content, with no contractual "copyright shield" to fall back on.
* `Integration Complexity`: Adapting open-source models to fit specific company datasets and infrastructure tools often requires rebuilding implementations from scratch, which can be more time-consuming than using a managed proprietary API.
* `Scalability Challenges`: Scaling open-source deployments requires significant investment in hardware and engineering, whereas proprietary models offer easy, managed scalability via API without vendor lock-in concerns for the underlying hardware. 

### Strategic and Support Factors

* `Enterprise Support`: Large organizations prefer proprietary vendors who offer Service Level Agreements (SLAs), safety protocols, and dedicated support contracts that open-source communities cannot guarantee.
* `Legal Indemnification`: Major proprietary providers (e.g., Microsoft, Google, OpenAI) offer contractual indemnification against copyright claims, a critical feature for regulated industries like finance and healthcare that open-source users must self-insure against.

## Ollama 
 It's a tool that lets you `run large language models(LLMs) on your own computer.` In simple terms Ollama makes it easy to download, run and manage AI models locally using very simple commands.

## Benifits of using Ollama:

1. Privacy and Data Control:
2. Lower Latency and offline access: 
3. Cost Predictability and potential savings:
4. Simple Installation and setup:
5. Pre-built Model Library:
6. Customization of Model:
7. Run, manage Models with simple commands

## Ways to use Ollama in your system:
1. CLI
2. Ollama Library
3. RestAPIs
4. Langchain Integration
5. Ollama application





