import streamlit as st
import json
import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

with open("sample-customer-reviews.json", "r") as file:
    reviews = json.load(file)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt Technique Tree of Thoughts
unified_prompt = PromptTemplate.from_template("""
You are an intelligent customer feedback analysis assistant. Given the following review, extract these details in plain text (no JSON format):

1. *Sentiment Classification*: Choose only one - Positive, Negative, or Neutral.
2. *Themes*: Extract main topics or features discussed in the review.Also include the product category.
3. *Response*: 
   - *Good Aspects*: Identify what the customer liked about the product or experience.
   - *Bad Aspects*: Identify any issues or concerns raised by the customer.
   - *Address Specific Concerns*: If there's a problem or concern, address it directly.
4. *Priority*: Choose only High, Medium, or Low based on sentiment, review date, and issue severity.

Review:
{review_json}

Output format:

Sentiment:  <Positive | Negative | Neutral>

Themes:  <comma-separated list of key themes>

Good Aspects of Product:  <list or summary of positive aspects>

Bad Aspects of Product:  <list or summary of negative aspects, if any>

Response:  <address specific concern(s) based on review>

Priority:  <High | Medium | Low> 

Priority Reason:  <brief explanation why>
""")

unified_chain = unified_prompt | llm | RunnableLambda(lambda out: out.content.strip())

st.subheader("DSCI Technical Assessment: AI-Powered Customer Review Analysis")
st.title("Customer Review Analysis")
st.text("We've got 50 customer reviews from DSCI ready for you to analyze—pick any one to get started!")

review_index = st.selectbox("Select a review to analyze,", range(len(reviews)))
selected_review = reviews[review_index]

if st.button("Analyze Review"):
    review_json = json.dumps(selected_review, indent=2)
    result = unified_chain.invoke({"review_json": review_json})
    st.subheader("Review Text")
    st.header(f"*{selected_review['review_text']}*")    
    st.subheader("Analysis Result:")
    st.text(result)

    sentiment = None
    themes = []
    for line in result.splitlines():
        if line.startswith("Sentiment:"):
            sentiment = line.split(":")[1].strip()
        elif line.startswith("Themes:"):
            themes = [t.strip().lower() for t in line.split(":")[1].split(",") if t.strip()]

    if sentiment:
        st.subheader("Sentiment Chart")
        fig, ax = plt.subplots()
        ax.bar([sentiment], [1], color='green' if sentiment == 'Positive' else 'red' if sentiment == 'Negative' else 'orange')
        st.pyplot(fig)

    if themes:
        st.subheader("Theme Word Cloud")
        wc = WordCloud(width=600, height=300, background_color="white").generate(" ".join(themes))
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)