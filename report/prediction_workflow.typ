#pagebreak()
= Prediction Workflow
1.  A user interacts with the `frontend_service` (Gradio UI), providing an image file, NIH tabular data, and sensor data.
2.  The `frontend_service` sends this data to the `api_gateway`.
3.  The `api_gateway` routes the request to the `/predict/` endpoint of the `disease_prediction_service`.
4.  The `disease_prediction_service`:
    a.  Receives the raw data.
    b.  Uses its loaded image feature extractor to get image features from the uploaded image.
    c.  Preprocesses the input NIH tabular data (encoding, scaling) using loaded artifacts.
    d.  Preprocesses the input sensor data (aggregation, scaling if applicable) using loaded artifacts or defined logic.
    e.  Concatenates/fuses these three sets of features.
    f.  Feeds the combined features into the loaded `AttentionFusionMLP` model.
    g.  Obtains raw output logits from the model.
    h.  Applies a sigmoid function to get probabilities for each disease class.
    i.  Optionally, applies a threshold (e.g., 0.5) to determine predicted labels.
    j.  Returns the list of diseases with their corresponding prediction probabilities (and/or binary predictions).
5.  The `api_gateway` forwards the response back to the `frontend_service`.
6.  The `frontend_service` displays the predictions to the user.

// Remember to create this diagram and place it in the images folder
// #figure(
//   image("images/prediction_workflow_diagram.png", width: 90%),
//   caption: [Prediction Workflow Diagram. Illustrates the sequence from user input in Gradio to the final prediction display.]
// )
// [PLACEHOLDER: Prediction Workflow Diagram. This diagram should illustrate the sequence from user input in Gradio to the final prediction display, showing data flow through API Gateway, Disease Prediction Service, and interaction with loaded model components.]