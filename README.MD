# UNH Manchester Course Internship Assistant Chatbot

## Overview
This project implements a specialized chatbot designed to assist UNH Manchester students with internship-related queries. The chatbot utilizes Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses about internship courses, requirements, and procedures.

## System Architecture
Below is the system architecture diagram showing the flow of data through our application:
![Alt text](images/flow.png)


The system consists of three main layers:
1. **Data Processing Layer**: Handles document ingestion and preparation
2. **Application Layer**: Manages user interactions and query processing
3. **Data Storage Layer**: Maintains chat history
## Technology Stack
- **Frontend**: Flask web interface
- **Backend**: Python with LangChain framework
- **LLM**: OpenAI GPT-4 Turbo for natural language processing
- **Vector Storage**: FAISS for efficient document retrieval
- **Database**: SQLite for chat history
- **Deployment**: Amazon EC2

## Key Features
- Domain-specific responses about UNH Manchester internship programs
- Course recommendation based on student situations
- Chat history tracking and analytics
- Document processing and intelligent retrieval
- Deployment on AWS cloud infrastructure

## Project Structure
```
project/
├── src/
│   ├── ragllm.py
│   ├── templates/
│   │   └── index.html
│   └── materials/
│       └── course_docs/
├── embeddings/
├── chat_history.db
├── requirements.txt
└── README.md
```

## Prerequisites
- Python 3.7+
- OpenAI API key
- AWS account for EC2 deployment

## Dependencies
```
flask
langchain
langchain-openai
faiss-cpu
python-dotenv
tqdm
openai
sqlite3
```

## Local Setup
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   python src/ragllm.py
   ```

## AWS EC2 Deployment Guide
Step 1: Launch EC2 Instance

Navigate to AWS EC2 Dashboard
Launch new instance:

Select Amazon Linux 2023 AMI
Choose t3.xlarge (4 vCPUs, 16 GiB RAM)
Configure network settings


Configure Security Group:
CopyInbound Rules:
- SSH (Port 22): Your IP
- HTTP (Port 80): Anywhere
- HTTPS (Port 443): Anywhere

Create and download key pair (.pem file)

Step 2: Connect to EC2 Instance
bashCopy# Set key pair permissions
chmod 400 your-key-pair.pem

### SSH into instance
ssh -i your-key-pair.pem ec2-user@your-ec2-public-dns
Step 3: Setup Environment
bashCopy# Update package manager
sudo yum update -y

### Install required packages
sudo yum install git python3 -y
Step 4: Deploy Application Code
bashCopy# Clone repository
git clone https://github.com/your-username/your-repo-name.git

### Navigate to project directory
cd your-repo-name
Step 5: Setup Dependencies
bashCopy# Create virtual environment
python3 -m venv venv

### Activate virtual environment
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt
Step 6: Configure Environment Variables
Create and configure .env file:
bashCopy# Create .env file
touch .env

### Add required environment variables
echo "OPENAI_API_KEY=your_api_key_here" >> .env
Step 7: Launch Application
bashCopy# Start Flask application
flask run --host=0.0.0.0 --port=80
Step 8: Verify Deployment

Access application through browser:
Copyhttp://your-ec2-public-dns

Verify all functionalities are working
Check for any error messages in the console
Monitor application logs

Troubleshooting Tips

Check security group settings if unable to connect
Verify all dependencies are installed correctly
Ensure environment variables are properly set
Review Flask logs for any error messages
Confirm port 80 is accessible and not blocked by firewall

Important Notes

Keep your .pem file secure and never share it
Regularly update security patches
Monitor EC2 instance metrics
Backup your environment configurations
Follow AWS best practices for production deployments

## Usage
1. Access the chatbot through your browser:
   - Local: `http://localhost:5000`
   - EC2: `http://your-ec2-ip:5000`
2. Type your internship-related questions in the chat interface
3. The chatbot will provide relevant information based on course materials

## Features
- Course recommendation based on student status (working, seeking internship, etc.)
- Information about course requirements and prerequisites
- Guidance on internship registration process
- CPT authorization assistance for international students
- Chat history tracking and analytics

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

