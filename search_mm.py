import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def connect_wxd():
    load_dotenv()

    wxd_endpoint = os.environ["WXD_URL"]
    wxd_username = os.environ["WXD_username"]
    wxd_pwd = os.environ["WXD_password"]
    es_cert_path = os.environ['wxd_cert_path']

    client = Elasticsearch(
        wxd_endpoint,
        ca_certs=es_cert_path,
        basic_auth=(wxd_username, wxd_pwd)
    )

    return client


def get_knn(query):
    KNN_E5={
        "field":"web_text_sentence_embedding",
        "query_vector_builder": {
        "text_embedding": {
            "model_id": "intfloat__multilingual-e5-large",
            "model_text": query,
        }
    },
    "k": 10,
    "num_candidates": 20
    }
    
    return KNN_E5


def wxd_search_basic(client, query, index):
    response = client.search(
        index=index,
        size=3,
        knn=get_knn(query),
    )
    return response


def make_prompt(query, hits):
    images = []
    contexts = ""
    for hit in hits:
        if "img" in hit["_source"]["title"]:
            images.append(hit["_source"]["image_blob"])
        elif "text" in hit["_source"]["title"]:
            contexts += (hit["_source"]["web_text"] + "\n====\n")
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "你是一位 AI 助手，請根據檢索到的 **文字片段** 與 **視覺資料**（如圖表、照片、示意圖等），為使用者問題生成清晰、準確且有幫助的答案。"}
            ]
        }
    ]

    user_content = [
        {"type": "text", "text": f"## 檢索到的文字：{contexts}"},
    ]

    for img in images:
        user_content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}})

    user_content.append({"type": "text", "text": f"## 使用者問題：：{query} \n## 你的回答："},)


    user_message = {
        "role": "user",
        "content": user_content
    }

    messages.append(user_message)
    return messages


def generate_answer(query, hits):
    import os
    from dotenv import load_dotenv
    from ibm_watsonx_ai.foundation_models import ModelInference

    load_dotenv()

    # --- Set your IBM watsonx credentials and config ---
    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    credentials = {
        "apikey": os.getenv("WATSONX_APIKEY"),
        "url": "https://us-south.ml.cloud.ibm.com"
    }


    params = {
        "decoding_method": "greedy",
        "max_new_tokens": 5000,
    }

    # --- Initialize watsonx model ---
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=os.getenv("WX_PROJECT_ID"),
        params=params
    )

    messages = make_prompt(query, hits)

    try:
        response = model.chat(messages=messages)
        answer = response["choices"][0]["message"]["content"].strip()
        # print("\n\n======= LLM ANSWER =========\n\n", answer)
    except Exception as e:
        print(f"Error processing image: {e}")

    return answer