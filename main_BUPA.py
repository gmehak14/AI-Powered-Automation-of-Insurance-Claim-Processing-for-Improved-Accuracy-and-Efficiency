# Import necessary libraries
import os, re
from flask import Flask, render_template, request
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
import json

# The Google API key is read automatically from the GOOGLE_API_KEY environment variable
FAISS_PATH = "/faiss"

# Flask App
app = Flask(__name__)

vectorstore = None
general_exclusion_list = [
    "HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", "pregnancy",
    "substance abuse", "self-inflicted injuries",
    "sexually transmitted diseases(std)", "pre-existing conditions"
]

def get_document_loader():
    loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    return loader.load()

def get_text_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_documents(documents)

def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    return FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

def get_retriever():
    return get_embeddings().as_retriever()

def get_claim_approval_context():
    db = get_embeddings()
    context = db.similarity_search("What are the documents required for claim approval?")
    return "".join([x.page_content for x in context])

def get_general_exclusion_context():
    db = get_embeddings()
    context = db.similarity_search("Give a list of all general exclusions")
    return "".join([x.page_content for x in context])

def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_bill_info(data, llm):
    prompt = """Act as an expert in extracting information from medical invoices.
    Extract "disease" and "expense amount". Return JSON = {"disease": "", "expense": ""}"""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"INVOICE DETAILS: {data}")
    ]
    
    response = llm.invoke(messages)
    cleaned = response.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except:
        print("JSON decode failed:", cleaned)
        return {"disease": "Unknown", "expense": None}




# ============= UPDATED REJECTION PROMPT =============
REJECTION_PROMPT = """
You are an AI assistant for verifying health insurance claims.
You must generate a structured claim rejection report based on the following data:

DOCUMENTS FOR CLAIM APPROVAL:
{claim_approval_context}

GENERAL EXCLUSION LIST:
{general_exclusion_context}

PATIENT INFO:
{patient_info}

MEDICAL BILL INFO:
{medical_bill_info}

The claim must be rejected because the patient has **{disease}**, which appears in the General Exclusion List.

---

Executive Summary
[Provide a brief summary explaining why the claim is rejected.]

Introduction
[Explain the purpose of the report and the verification steps.]

Claim Details
[Describe the submitted claim details.]

Claim Description
[Short description of the claim.]

Document Verification
[Which documents were submitted and verified.]

Document Summary
[Summarize the medical findings, disease, and rejection reason.]

FINAL DECISION:
The claim is **REJECTED** because {disease} is excluded under policy rules.
"""

# =====================================================



# ============= ACCEPTANCE PROMPT (Original) =============
ACCEPTANCE_PROMPT = """
You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. Use the following guidelines for your analysis.

1. Verify if the patient has provided all necessary documents.
2. If any disease is in the general exclusions list → reject.

DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}
EXCLUSION LIST : {general_exclusion_context}
PATIENT INFO : {patient_info}
MEDICAL BILL : {medical_bill_info}

Write whether INFORMATION and EXCLUSION are TRUE or FALSE.
Reject if any is FALSE.
If accepted, maximum amount approved = {max_amount}

Executive Summary
[Summary]

Introduction
[Aim]

Claim Details
[Details]

Claim Description
[Short description]

Document Verification
[Docs submitted or missing]

Document Summary
[Summary of everything]
"""
# =====================================================




def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template, threshold=0.4):

    vectorizer = CountVectorizer()
    patient_vec = vectorizer.fit_transform([claim_reason])

    for disease in general_exclusion_list:
        dis_vec = vectorizer.transform([disease])
        similarity = cosine_similarity(patient_vec, dis_vec)[0][0]

        if similarity > threshold:
            return REJECTION_PROMPT  

    return ACCEPTANCE_PROMPT




@app.route('/')
def index():
    return render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def msg():
    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']
    medical_bill = request.files['medical_bill']
    total_claim_amount = request.form['total_claim_amount']
    description = request.form['description']

    bill_data = get_file_content(medical_bill)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    bill_info = get_bill_info(bill_data, llm)

    # Expense > claim → reject
    if bill_info["expense"] and int(bill_info["expense"]) < int(total_claim_amount):
        msg = "The claimed amount is more than the billed amount. Claim Rejected."
        return render_template("result.html", output=msg)

    # Expense valid → continue
    patient_info = (
        f"Name: {name}\nAddress: {address}\nClaim Type: {claim_type}\n"
        f"Claim Reason: {claim_reason}\nMedical Facility: {medical_facility}\n"
        f"Date: {date}\nClaim Amount: {total_claim_amount}\nDescription: {description}"
    )

    medical_bill_info = f"Bill Extracted: {bill_data}"

    validated_prompt = check_claim_rejection(
        bill_info["disease"], general_exclusion_list, ACCEPTANCE_PROMPT
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "claim_approval_context",
            "general_exclusion_context",
            "patient_info",
            "medical_bill_info",
            "max_amount",
            "disease"
        ],
        template=validated_prompt
    )

    llmchain = LLMChain(llm=llm, prompt=prompt_template)

    output = llmchain.run({
        "claim_approval_context": get_claim_approval_context(),
        "general_exclusion_context": get_general_exclusion_context(),
        "patient_info": patient_info,
        "medical_bill_info": medical_bill_info,
        "max_amount": total_claim_amount,
        "disease": bill_info["disease"]
    })

    output = output.replace("\n", "<br>")

    return render_template("result.html", output=output)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)


