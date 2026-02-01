import gradio as gr
from src import predict_area

with gr.Blocks(title="Площадь зданий со спутника") as demo:
    gr.Markdown("# 🛰️ Расчет площади зданий по спутниковому снимку")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Загрузите спутниковый снимок", type="numpy")
            auto_scale = gr.Checkbox(label="Авто-определение масштаба", value=True)
            manual_scale = gr.Slider(0.1, 2.0, 0.3, step=0.01, label="Ручной масштаб, м/пикс")
        
        with gr.Column():
            img_out = gr.Image(label="Маска зданий", type="numpy")
            area_text = gr.Textbox(label="Площадь всех зданий, м²")
            scale_text = gr.Textbox(label="Установленный масштаб, м/пикс")
    
    btn = gr.Button("Рассчитать")
    btn.click(
        predict_area,
        inputs=[img_in, auto_scale, manual_scale],
        outputs=[img_out, area_text, scale_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)