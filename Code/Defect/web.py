import os
import cv2
import gradio as gr
import detect
import arduinoControl
import camera
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_GERBER_PATH = None
css = """
.no-buttons [class*="button"], 
.no-buttons [id*="button"],
.no-buttons div[style*="position: absolute"] button,
.no-buttons svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
"""

INSPECTION_SHARED_STATE = {
    "has_new_data": False,
    "img_path": None,
    "report_text": ""
}

def update_gerber_path(file_obj):
    if file_obj is not None:
        return file_obj.name
    return None

def on_gerber_change(file_obj):
    global CURRENT_GERBER_PATH
    if file_obj is not None:
        CURRENT_GERBER_PATH = file_obj.name
        print(f"📂 Gerber успешно загружен в систему. Путь: {CURRENT_GERBER_PATH}")
    else:
        CURRENT_GERBER_PATH = None
    return file_obj

def check_for_background_updates(current_img, current_report):
    global INSPECTION_SHARED_STATE
    if INSPECTION_SHARED_STATE["has_new_data"]:
        INSPECTION_SHARED_STATE["has_new_data"] = False
        print("🔄 Интерфейс Gradio обнаружил новые данные и обновляет UI!")
        return INSPECTION_SHARED_STATE["img_path"], INSPECTION_SHARED_STATE["report_text"]
    return gr.update(), gr.update()

def trigger_auto_inspection():
    global CURRENT_GERBER_PATH, INSPECTION_SHARED_STATE
    if CURRENT_GERBER_PATH is None:
        print("❌ Критическая ошибка: Физический путь к Gerber файлу пуст!")
        return
        
    print(f"🚀 Запуск анализа. Gerber: {CURRENT_GERBER_PATH}, Панорама: {detect.output_skleika}")
    
    try:
        res_img_rgb, report_txt = detect.run_inspection(
            CURRENT_GERBER_PATH,
            detect.output_skleika,
            int(cfg_max_working_side.value) if hasattr(cfg_max_working_side, 'value') else 2200,
            int(cfg_filter_d.value) if hasattr(cfg_filter_d, 'value') else 9,
            int(cfg_sigma_color.value) if hasattr(cfg_sigma_color, 'value') else 75,
            int(cfg_sigma_space.value) if hasattr(cfg_sigma_space, 'value') else 75,
            int(cfg_block_size.value) if hasattr(cfg_block_size, 'value') else 59,
            int(cfg_c_val.value) if hasattr(cfg_c_val, 'value') else -14,
            str(cfg_noise_method.value) if hasattr(cfg_noise_method, 'value') else "Морфологическое открытие (Быстро)",
            int(cfg_morph_size.value) if hasattr(cfg_morph_size, 'value') else 4,
            int(cfg_min_noise_area.value) if hasattr(cfg_min_noise_area, 'value') else 250,
            int(cfg_min_defect_area.value) if hasattr(cfg_min_defect_area, 'value') else 200,
            int(cfg_large_defect_area.value) if hasattr(cfg_large_defect_area, 'value') else 800
        )
        
        debug_dir = "./debuging"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        debug_img_path = os.path.join(debug_dir, "inspection_result.png")
        debug_txt_path = os.path.join(debug_dir, "inspection_report.txt")
        
        if res_img_rgb is not None:
            res_img_bgr = cv2.cvtColor(res_img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(debug_img_path, res_img_bgr)
            print(f"Изображение анализа успешно сохранено в: {debug_img_path}")
        else:
            print("Внимание: Матрица изображения пуста.")
            return
            
        with open(debug_txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        print(f"Текстовый отчет успешно сохранен в: {debug_txt_path}")

        INSPECTION_SHARED_STATE["img_path"] = debug_img_path
        INSPECTION_SHARED_STATE["report_text"] = report_txt
        INSPECTION_SHARED_STATE["has_new_data"] = True
        print("📈 Данные подготовлены для отображения в Gradio.")

    except Exception as e:
        print(f"Ошибка внутри триггера автоматического анализа: {e}")

# Построение интерфейса
with gr.Blocks(title="Травилка") as demo:
    with gr.Row():
        gr.Markdown("# Система определения печатных плат при травлении")
        gr.Image(value="Web/Логотип_black.svg", show_label=False, container=False, height=80, interactive=False, elem_classes="no-buttons", buttons=[])
        gr.Image(value="Web/БВ Лого.svg", show_label=False, container=False, height=80, interactive=False, elem_classes="no-buttons", buttons=[])

    with gr.Tabs():
        with gr.TabItem("Главная"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Выберите файлы в проводнике")
                    input_gbr_file = gr.File(label="Загрузить Gerber файл (.gbr)", file_count="single")
                    gerber_path_state = gr.State(None)
                    
                    input_gbr_file.change(fn=update_gerber_path, inputs=[input_gbr_file], outputs=[gerber_path_state])
                    input_gbr_file.change(fn=on_gerber_change, inputs=[input_gbr_file], outputs=[])

                    input_pcb_file = gr.Image(label="Загрузить фото платы", type="filepath")
                    
                    with gr.Row():
                        btn_start_work = gr.Button("Начать работу", variant="secondary")
                        btn_run_main = gr.Button("Работа (тестирование)", variant="primary")
                        btn_open_cam = gr.Button("Открыть камеру", variant="secondary")
                    status_output = gr.Textbox(label="Статус системы / Лог пустой функции", placeholder="Здесь будет лог...")
                    
                with gr.Column():
                    gr.Markdown("### Результат анализа")
                    result_image = gr.Image(label="Карта дефектов")
                    result_report = gr.Textbox(label="Отчет", lines=6)

        with gr.TabItem("Настройки"):
            gr.Markdown("### Тонкая настройка алгоритмов обработки изображений")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Разрешение и геометрия")
                    cfg_max_working_side = gr.Number(label="max_working_side", value=detect.DEFAULT_CONFIG["max_working_side"], precision=0)
                    gr.Markdown("#### Билинейная фильтрация (Bilateral Filter)")
                    cfg_filter_d = gr.Slider(label="filter_d", minimum=1, maximum=25, step=1, value=detect.DEFAULT_CONFIG["filter_d"])
                    cfg_sigma_color = gr.Slider(label="sigma_color", minimum=10, maximum=200, step=5, value=detect.DEFAULT_CONFIG["sigma_color"])
                    cfg_sigma_space = gr.Slider(label="sigma_space", minimum=10, maximum=200, step=5, value=detect.DEFAULT_CONFIG["sigma_space"])

                with gr.Column():
                    gr.Markdown("#### Бинаризация и Очистка шума")
                    cfg_block_size = gr.Slider(label="block_size", minimum=3, maximum=151, step=2, value=detect.DEFAULT_CONFIG["block_size"])
                    cfg_c_val = gr.Slider(label="c_val", minimum=-50, maximum=50, step=1, value=detect.DEFAULT_CONFIG["c_val"])
                    cfg_noise_method = gr.Dropdown(label="noise_method", choices=["Без очистки", "Морфологическое открытие (Быстро)", "Фильтрация по площади (Чисто)"], value=detect.DEFAULT_CONFIG["noise_method"])
                    cfg_morph_size = gr.Slider(label="morph_size", minimum=1, maximum=15, step=1, value=detect.DEFAULT_CONFIG["morph_size"])
                    cfg_min_noise_area = gr.Number(label="min_noise_area", value=detect.DEFAULT_CONFIG["min_noise_area"], precision=0)

                with gr.Column():
                    gr.Markdown("#### Порог площади дефектов")
                    cfg_min_defect_area = gr.Number(label="min_defect_area", value=detect.DEFAULT_CONFIG["min_defect_area"], precision=0)
                    cfg_large_defect_area = gr.Number(label="large_defect_area (Критический порог)", value=detect.DEFAULT_CONFIG["large_defect_area"], precision=0)

    auto_refresh_timer = gr.Timer(value=1.0)
    auto_refresh_timer.tick(fn=check_for_background_updates, inputs=[result_image, result_report], outputs=[result_image, result_report])

    btn_start_work.click(fn=arduinoControl.Start_Work_Routine, inputs=[], outputs=[status_output])
    btn_open_cam.click(fn=camera.CameraInit, inputs=[], outputs=[])
    
    btn_run_main.click(
        fn=detect.run_inspection,
        inputs=[
            input_gbr_file, input_pcb_file, cfg_max_working_side, cfg_filter_d,
            cfg_sigma_color, cfg_sigma_space, cfg_block_size, cfg_c_val,
            cfg_noise_method, cfg_morph_size, cfg_min_noise_area,
            cfg_min_defect_area, cfg_large_defect_area
        ],
        outputs=[result_image, result_report]
    )

def main():
    demo.launch(share=False, css=css)
    camera.CameraInit()

if __name__ == "__main__":
    main()