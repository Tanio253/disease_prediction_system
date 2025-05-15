import gradio as gr
import httpx
import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (e.g., API URL)
load_dotenv()
PREDICTION_API_URL = os.getenv("DISEASE_PREDICTION_SERVICE_URL", "http://localhost:8004/predict/")
# Fallback to localhost if running locally without docker-compose env var set.
# The docker-compose.yml sets DISEASE_PREDICTION_SERVICE_URL to http://disease_prediction_service:8004/predict/

# Define the list of disease classes - this MUST match the output order of the prediction service
# Ideally, this comes from a shared config or the prediction service itself via an info endpoint
ALL_DISEASE_CLASSES_FRONTEND = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "No Finding"
]

async def predict_disease(image_filepath, patient_age, patient_gender, view_position, sensor_data_filepath):
    """
    Makes a request to the backend disease prediction service.
    """
    status_message = "Processing..."
    output_df = pd.DataFrame(columns=["Disease", "Probability"]) # Default empty DataFrame

    if image_filepath is None:
        return pd.DataFrame({"Disease": ["Error"], "Probability": ["No image uploaded."]}), "Error: Please upload an X-ray image."

    files_to_send = {}
    data_to_send = {
        "patient_age": int(patient_age), # Ensure age is int
        "patient_gender": patient_gender,
        "view_position": view_position,
    }

    try:
        # Prepare image file
        files_to_send["image_file"] = (os.path.basename(image_filepath), open(image_filepath, "rb"), "image/png") # Adjust content type if needed

        # Prepare optional sensor data file
        if sensor_data_filepath is not None:
            files_to_send["sensor_data_csv"] = (os.path.basename(sensor_data_filepath), open(sensor_data_filepath, "rb"), "text/csv")

        logger.info(f"Sending request to {PREDICTION_API_URL} with data: {data_to_send.keys()} and files: {files_to_send.keys()}")

        async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout for potentially slow model loading/inference
            response = await client.post(
                PREDICTION_API_URL,
                data=data_to_send,
                files=files_to_send
            )

        logger.info(f"Received response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            predictions = response_data.get("predictions", [])
            
            if predictions:
                # Convert list of dicts to DataFrame for Gradio
                df_data = []
                for p in predictions:
                    df_data.append({"Disease": p["disease_name"], "Probability": f"{p['probability']:.4f}"})
                output_df = pd.DataFrame(df_data)
                status_message = "Prediction successful."
                if response_data.get("errors"):
                    status_message += f" Warnings: {', '.join(response_data['errors'])}"
            else:
                status_message = "Prediction successful, but no prediction data returned."
                if response_data.get("errors"):
                    status_message = f"Error: {', '.join(response_data['errors'])}"

        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                error_detail = response.text
            status_message = f"Error: Backend returned status {response.status_code}. Detail: {error_detail}"
            logger.error(status_message)

    except httpx.ConnectError as e:
        status_message = f"Error: Could not connect to the prediction service at {PREDICTION_API_URL}. Is it running? Detail: {str(e)}"
        logger.error(status_message, exc_info=True)
    except httpx.ReadTimeout:
        status_message = f"Error: Request to prediction service timed out after 120s. The model might be taking too long to predict."
        logger.error(status_message, exc_info=True)
    except Exception as e:
        status_message = f"An unexpected error occurred: {str(e)}"
        logger.error(status_message, exc_info=True)
    finally:
        # Close file objects if they were opened
        if "image_file" in files_to_send and files_to_send["image_file"]:
            files_to_send["image_file"][1].close()
        if "sensor_data_csv" in files_to_send and files_to_send["sensor_data_csv"]:
            files_to_send["sensor_data_csv"][1].close()


    return output_df, status_message


# Define Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multi-modal Disease Prediction System")
    gr.Markdown("Upload an X-ray image, provide patient details, and optionally sensor data to predict potential diseases.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Data")
            image_input = gr.Image(type="filepath", label="Upload X-ray Image", sources=["upload"])
            age_input = gr.Number(label="Patient Age", minimum=0, maximum=120, step=1, value=50)
            gender_input = gr.Radio(label="Patient Gender", choices=["M", "F", "O"], value="M") # 'O' for Other/Unknown
            view_pos_input = gr.Radio(label="X-Ray View Position", choices=["PA", "AP", "LL", "RL", "XX"], value="PA") # Added 'XX'
            sensor_file_input = gr.File(label="Sensor Data CSV (Optional)", type="filepath", file_types=[".csv"])
            
            predict_button = gr.Button("Predict Diseases", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Prediction Results")
            status_output = gr.Textbox(label="Status / Messages", lines=2, interactive=False)
            predictions_output_df = gr.Dataframe(
                headers=["Disease", "Probability"],
                label="Disease Predictions",
                col_count=(2, "fixed"),
                wrap=True,
                max_rows=15 # For the 15 classes
            )

    predict_button.click(
        fn=predict_disease,
        inputs=[image_input, age_input, gender_input, view_pos_input, sensor_file_input],
        outputs=[predictions_output_df, status_output]
    )

    gr.Markdown("---")
    gr.Markdown("### Notes:")
    gr.Markdown("- Ensure all backend services (especially `disease_prediction_service`) are running.")
    gr.Markdown("- The `Sensor Data CSV` should have columns like 'Timestamp', 'HeartRate_bpm', 'SpO2_percent', etc., as expected by the backend.")
    gr.Markdown("- **Patience Please:** The first prediction after service startup might be slower due to model loading in the backend.")


if __name__ == "__main__":
    logger.info(f"Starting Gradio frontend. Prediction API URL: {PREDICTION_API_URL}")
    # For local testing without Docker, PREDICTION_API_URL might be http://localhost:8004/predict/
    # When run with Docker Compose, it will use the service name: http://disease_prediction_service:8004/predict/
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    # share=True would create a public link if you need to test externally (requires Gradio account or flags)