import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIG ---
FOLDER_PATH = "logi_chatbot.txt"  # File chứa dữ liệu
FAISS_INDEX_PATH = "faiss_index"  # Nơi lưu FAISS index
GOOGLE_API_KEY = "AIzaSyB5Rv1wk7qiEsf1P8NSMYXrAZVoo7D3_H0"
MODEL_NAME = "gemini-1.5-pro"  # Sử dụng model mạnh hơn

# --- STEP 1: Load file .txt ---
def load_text_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# --- STEP 2: Tạo FAISS Index ---
def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# --- STEP 3: Tìm kiếm với FAISS + Hỏi Gemini ---
def query_with_gemini(user_query, k=5):
    # Load FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Tìm kiếm nội dung liên quan
    related_docs = vectorstore.similarity_search(user_query, k=k)
    context = "\n\n".join([doc.page_content for doc in related_docs])

    # Kiểm tra nếu FAISS không tìm thấy nội dung phù hợp
    if not context.strip():
        return "❌ Không tìm thấy thông tin phù hợp trong dữ liệu!"

    # Kết nối Gemini API
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # Tạo prompt gửi đến Gemini
    full_prompt = f"""
    Dựa vào các thông tin sau đây, hãy trả lời câu hỏi một cách chính xác:

    ---Thông tin---
    {context}

    ---Câu hỏi---
    {user_query}
    """
    # Gửi đến Gemini
    response = llm.invoke(full_prompt)
    return response.content if response else "❌ Lỗi: Gemini không phản hồi!"

# --- MAIN ---
if __name__ == "__main__":
    # Bước 1: Load file và tạo FAISS (chạy 1 lần khi có dữ liệu mới)
    documents = load_text_file(FOLDER_PATH)
    create_faiss_index(documents)

    # Bước 2: Hỏi đáp qua FAISS + Gemini
    while True:
        query = input("\nNhập câu hỏi (hoặc 'exit' để thoát): ")
        if query.lower() == "exit":
            break
        answer = query_with_gemini(query)
        print("\n🤖 Chatbot trả lời:", answer)
