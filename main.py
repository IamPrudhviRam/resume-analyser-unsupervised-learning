import os
import ExtractText as extractText
from textblob import TextBlob as tb

dir_path="resume_samples"

global documents
documents=[]


for f in os.listdir(dir_path):
    document_dict={}
    bagOfWords_dict={}
    file_path = os.path.join(dir_path, f)
    base = os.path.basename(file_path)
    filename = os.path.splitext(base)[0]
    filename=filename.replace(" ","_")
    if f.lower().endswith('.docx'):
        text = extractText.docx_to_text(file_path)
        # print("docx",text)

    elif f.lower().endswith('.pdf'):
        text = extractText.pdf_to_text(file_path)
        text = text.replace('\n', ' ').replace('\r', '')
        # print("pdf", text)
    documents.append(text)






