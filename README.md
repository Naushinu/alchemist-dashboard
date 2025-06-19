# Mapping the Soul of *The Alchemist*

An interactive NLP dashboard analyzing sentiment and emotion in Paulo Coelho’s *The Alchemist*, combining computational analysis with literary insight.

---

## Project Summary

**Mapping the Soul of *The Alchemist*** is a digital humanities initiative project built with **Python**, **NLP**, and **data visualization** tools to uncover the emotional and geographic journey of Santiago, the protagonist of *The Alchemist*. Using modern sentiment analysis libraries and visualization frameworks, this interactive dashboard brings literature and data together in a compelling user-facing product.

Developed as part of a capstone humanities seminar at WPI, this project demonstrates how literature can be explored through a computational lens—blending critical thinking with technical execution.

---

## Objectives

- Analyze sentiment at the sentence level using **VADER**
- Detect emotions using both **Text2Emotion** (coarse) and **NRC Emotion Lexicon** (nuanced)
- Visualize **rolling sentiment trends** and **segment-based emotion shifts**
- Geospatially map the character's journey across North Africa using **Plotly**
- Engage with questions of literary translation, narrative voice, and emotional interpretation

---

## Technologies & Tools

- **Python** (Pandas, Numpy, Matplotlib, Seaborn)
- **Natural Language Processing** (VADER, Text2Emotion, NLTK, NRCLex)
- **Data Visualization** (Plotly, Streamlit, Matplotlib)
- **Geospatial Mapping** (Plotly's `scattergeo`)
- **Streamlit** (for interactive dashboard deployment)

---

## Features

- **Sentiment Trends**: Visualize positive/negative sentiment over time and across book segments
- **Emotion Detection**: Compare coarse and fine-grained emotions detected sentence by sentence
- **Narrative Mapping**: Interactive map showing Santiago’s physical journey
- **Translation Note**: Acknowledge limitations and nuance of analyzing translated literature

---

## Getting Started

Clone the repository and run the dashboard locally:

```bash
git clone https://github.com/Naushinu/alchemist-dashboard.git
cd alchemist-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

# Acknowledgments

- Paulo Coelho: Author of The Alchemist 
- WPI HU 3900 Seminar: Support and guidance throughout this digital humanities capstone
