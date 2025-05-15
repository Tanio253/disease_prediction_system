#set document(title: "AI-Powered Multi-Modal Disease Prediction System", author: "Thanh Tran") 
#set heading(numbering: "1.")
#show outline: set text(size: 10pt) // 

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2cm, right: 2cm),
)
#set text(font: "New Computer Modern", size: 11pt, lang: "en") 

#align(center)[
  #text(weight: "bold", size: 1.8em)[AI-Powered Multi-Modal Disease Prediction System]
  #v(1em)
  #text(size: 1.2em)[Report]
  #v(0.5em)
  Thanh Tran
  #v(0.2em)
  May 15, 2025 
  #v(2em)
]

#block(inset: (left: 1cm, right: 1cm))[
  #text(weight: "bold")[Abstract:]
  This report details the system design and implementation of an advanced AI-powered disease prediction application. The system leverages comprehensive health data, including medical imagery (Chest X-rays from the NIH dataset), tabular patient metadata, and simulated physiological sensor data, to predict the likelihood of 14 common thoracic diseases. Architected as a suite of containerized microservices using Docker, the system employs Python, FastAPI, PyTorch, PostgreSQL, MinIO, and Gradio to deliver an end-to-end solution from data ingestion and preprocessing to model training and interactive prediction. Key features include a multi-modal fusion model, automated data processing pipelines triggered via background tasks, and a user-friendly interface for inference.
]

#outline()
#pagebreak()
#include "introduction.typ"
#include "system_overview.typ"
#include "microservice_details.typ"
#include "data_management.typ"
#include "ml_pipeline.typ"
#include "prediction_workflow.typ"
#include "technology_stack.typ"
#include "deployment.typ"
#include "conclusion.typ"