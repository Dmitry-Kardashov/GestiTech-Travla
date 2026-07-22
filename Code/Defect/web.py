# -*- coding: utf-8 -*-
import os
import sys
import cv2
import gradio as gr
import detect
import arduinoControl
import camera
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Принудительно переключаем stdout/stderr на UTF-8, чтобы кириллица в логах
# не превращалась в "кракозябры" при другой локали системы/консоли.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# КЛЮЧЕВОЙ ФИКС: когда этот файл запускается напрямую (python web.py),
# Python регистрирует его в sys.modules под именем '__main__', а НЕ 'web'.
# А arduinoControl._finalize_and_stitch() ищет именно
# sys.modules.get('web'), чтобы после склейки панорамы вызвать
# web.trigger_auto_inspection(). Из-за этого поиск возвращал None,
# и весь автоматический запуск анализа дефектов после снимков молча
# не срабатывал. Регистрируем себя под именем 'web' явно.
sys.modules.setdefault('web', sys.modules[__name__])

CURRENT_GERBER_PATH = None

# Живое хранилище текущих значений вкладки "Настройки".
# ВАЖНО: gr.Slider/gr.Number/gr.Dropdown.value хранит только НАЧАЛЬНОЕ значение,
# заданное при создании виджета, и НЕ обновляется, когда пользователь двигает
# ползунок в интерфейсе. Поэтому раньше trigger_auto_inspection() (запускаемый
# автоматически после склейки панорамы) всегда использовал значения по
# умолчанию, даже если пользователь поменял их на вкладке "Настройки".
# Здесь мы храним актуальные значения и обновляем их через .change()-обработчики.
CURRENT_CONFIG = dict(detect.DEFAULT_CONFIG)


def make_config_updater(key):
    """Возвращает обработчик, который кладет новое значение виджета в CURRENT_CONFIG."""
    def _update(value):
        CURRENT_CONFIG[key] = value
        return value
    return _update
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
        cfg = CURRENT_CONFIG
        res_img_rgb, report_txt = detect.run_inspection(
            CURRENT_GERBER_PATH,
            detect.output_skleika,
            int(cfg["max_working_side"]),
            int(cfg["filter_d"]),
            int(cfg["sigma_color"]),
            int(cfg["sigma_space"]),
            int(cfg["block_size"]),
            int(cfg["c_val"]),
            str(cfg["noise_method"]),
            int(cfg["morph_size"]),
            int(cfg["min_noise_area"]),
            int(cfg["min_defect_area"]),
            int(cfg["large_defect_area"])
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
                        btn_start_work = gr.Button("Начать работу", variant="primary")
                        btn_run_main = gr.Button("Работа (тестирование)", variant="secondary")
                        btn_open_cam = gr.Button("Открыть камеру", variant="secondary")
                    with gr.Row():
                        btn_stop_motor = gr.Button("Остановить мотор", variant="stop")
                        btn_lower_board = gr.Button("Опустить плату", variant="secondary")
                    with gr.Row():
                        btn_calibrate = gr.Button("Калибровка двигателей", variant="secondary")
                        
                    status_output = gr.Textbox(label="Статус системы / Лог пустой функции", placeholder="Здесь будет лог...")
                    
                with gr.Column():
                    gr.Markdown("### Результат анализа")
                    result_image = gr.Image(label="Карта дефектов")
                    result_report = gr.Textbox(label="Отчет", lines=6)

        with gr.TabItem("Настройки"):
            gr.Markdown("### Управление железом и алгоритмами")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Позиции ABS_MOVE (мотор)")
                    cfg_abs_positions = gr.Textbox(
                        label="Позиции ABS_MOVE по очереди (через запятую)",
                        value=",".join(str(p) for p in arduinoControl.ABS_MOVE_POSITIONS),
                        placeholder="3000,4000,5000,5200"
                    )

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

    # Синхронизация вкладки "Настройки" с CURRENT_CONFIG: любое изменение виджета
    # сразу же попадает в словарь, которым пользуется автоматический анализ.
    cfg_max_working_side.change(fn=make_config_updater("max_working_side"), inputs=[cfg_max_working_side], outputs=[])
    cfg_filter_d.change(fn=make_config_updater("filter_d"), inputs=[cfg_filter_d], outputs=[])
    cfg_sigma_color.change(fn=make_config_updater("sigma_color"), inputs=[cfg_sigma_color], outputs=[])
    cfg_sigma_space.change(fn=make_config_updater("sigma_space"), inputs=[cfg_sigma_space], outputs=[])
    cfg_block_size.change(fn=make_config_updater("block_size"), inputs=[cfg_block_size], outputs=[])
    cfg_c_val.change(fn=make_config_updater("c_val"), inputs=[cfg_c_val], outputs=[])
    cfg_noise_method.change(fn=make_config_updater("noise_method"), inputs=[cfg_noise_method], outputs=[])
    cfg_morph_size.change(fn=make_config_updater("morph_size"), inputs=[cfg_morph_size], outputs=[])
    cfg_min_noise_area.change(fn=make_config_updater("min_noise_area"), inputs=[cfg_min_noise_area], outputs=[])
    cfg_min_defect_area.change(fn=make_config_updater("min_defect_area"), inputs=[cfg_min_defect_area], outputs=[])
    cfg_large_defect_area.change(fn=make_config_updater("large_defect_area"), inputs=[cfg_large_defect_area], outputs=[])

    auto_refresh_timer = gr.Timer(value=1.0)
    auto_refresh_timer.tick(fn=check_for_background_updates, inputs=[result_image, result_report], outputs=[result_image, result_report])

    # Обработчики кнопок
    btn_start_work.click(
        fn=arduinoControl.Start_Work_Routine, 
        inputs=[cfg_abs_positions], 
        outputs=[status_output]
    )
    btn_stop_motor.click(fn=arduinoControl.Stop_Motor, inputs=[], outputs=[status_output])
    btn_lower_board.click(fn=arduinoControl.Lower_Board, inputs=[], outputs=[status_output])
    btn_calibrate.click(fn=arduinoControl.Motor_Calibrate, inputs=[], outputs=[status_output])
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
    # ВНИМАНИЕ: demo.launch() блокирует выполнение, пока сервер работает,
    # поэтому строка camera.CameraInit() после него раньше никогда не
    # выполнялась (мертвый код). Камера уже открывается по кнопке
    # "Открыть камеру" (btn_open_cam) - отдельный вызов здесь не нужен.
    demo.launch(share=False, css=css)

if __name__ == "__main__":
    main()