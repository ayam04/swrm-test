import os
import gradio as gr
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

llm = GroqModel(api_key=API_KEY)
allowed_models = llm.allowed_models

conversation = MaxSystemContextConversation()

def load_model(selected_model):
    return GroqModel(api_key=API_KEY, name=selected_model)

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    
    pdf_reader = PdfReader(pdf_file.name)
    text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
    return text

pdf_content = ""

def update_pdf_content(pdf_file):
    global pdf_content
    pdf_content = extract_text_from_pdf(pdf_file) if pdf_file else ""
    return "PDF uploaded and processed successfully." if pdf_content else "No PDF uploaded or PDF is empty."

def converse(message: str, history, system_context, model_name):
    global pdf_content, conversation
    
    llm = load_model(model_name)
    
    full_context = f"{system_context}\n\nPDF Content:\n{pdf_content}" if pdf_content else system_context
    conversation = MaxSystemContextConversation()
    conversation.system_context = SystemMessage(content=full_context)
    
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    result = str(agent.exec(message.strip()))
    
    return result

with gr.Blocks(theme=gr.themes.Soft(spacing_size='sm')) as demo:
    gr.Markdown("# Groq Chat with PDF Context")
    
    chat_interface = gr.ChatInterface(
        additional_inputs=[
            gr.Textbox(label="System Context", placeholder="Enter any specific instructions or context"),
            gr.Dropdown(label="Model Name", choices=allowed_models, value=allowed_models[0])
        ],
        fn=converse,
        title="Groq Chat",
        description="Chat with Groq models using optional PDF context",
        show_progress='minimal',
        fill_height=True
    )
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF (Optional)", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="PDF Status", interactive=False)
    
    pdf_upload.upload(fn=update_pdf_content, inputs=[pdf_upload], outputs=[pdf_status])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)