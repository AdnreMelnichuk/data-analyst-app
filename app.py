import streamlit as st
import pandas as pd
import json
import requests

API_KEY = "ключ"

def analyze_with_llm(df, instruction):
    forbidden = [
        "ignore previous", "forget", "disregard", "system prompt", "jailbreak",
        "игнорируй", "забудь", "притворись", "ты теперь", "новая инструкция",
        "игнорируй предыдущие", "забудь всё", "отключи", "обойди"
    ]
    for word in forbidden:
        if word.lower() in instruction.lower():
            return "Обнаружена подозрительная инструкция. Попробуйте другой запрос."

    sample = df.head(5).to_string()
    stats = df.describe(include='all').to_string()
    columns = list(df.columns)
    shape = df.shape

    prompt = f"""Ты аналитик данных. Тебе дан датасет.

Размер датасета: {shape[0]} строк, {shape[1]} столбцов
Колонки: {columns}

Первые 5 строк:
{sample}

Статистика:
{stats}

Инструкция от пользователя: {instruction}

Проведи полный анализ датасета:
1. Опиши что это за данные
2. Ключевые метрики и цифры
3. Интересные закономерности и инсайты
4. Выводы и рекомендации

Отвечай на русском языке, структурированно."""

    response = requests.post(
        url="https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
        }
    )

    return response.json()["choices"][0]["message"]["content"]


st.set_page_config(page_title="Аналитик данных")
st.title("AI Аналитик данных")
st.write("Загрузите CSV-файл и получите анализ от ИИ")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
instruction = st.text_area(
    "Инструкция (необязательно)",
    placeholder="Например: обрати внимание на продажи по регионам, найди аномалии..."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Данные")
    st.dataframe(df.head(10))
    st.write(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")

    if st.button("Анализировать"):
        with st.spinner("ИИ анализирует данные..."):
            result = analyze_with_llm(df, instruction)
        st.subheader("Результат анализа")
        st.markdown(result)