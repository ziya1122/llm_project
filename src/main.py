import gradio as gr
from PIL import Image
import os
from multimodal_rag_system import MultimodalRAGSystem

# Initialize the system
system = MultimodalRAGSystem()

# Define chatbot function
def chatbot_interface(user_query, user_image=None):
    if user_image is not None:
        # Convert numpy array to PIL Image
        user_image = Image.fromarray(user_image)

        # ✅ FIXED PATH (dynamic + cross-platform)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(base_dir, exist_ok=True)

        user_image_path = os.path.join(base_dir, "user_input_image.jpg")

        # Save image
        user_image.save(user_image_path)
    else:
        user_image_path = None

    # Process query
    response = system.process_query(user_query, query_image_path=user_image_path)
    return response


# Gradio UI
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(lines=5, label="User Query"),
        gr.Image(label="Upload Medical Image")
    ],
    outputs=gr.Textbox(),
    title="Multimodal Medical Assistant",
    description="Ask medical-related questions and upload relevant medical images for analysis."
)

# Run app
if __name__ == "__main__":
    interface.launch()