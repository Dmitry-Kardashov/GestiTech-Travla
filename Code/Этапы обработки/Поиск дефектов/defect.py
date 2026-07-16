import cv2
import gradio as gr
import numpy as np

# ==========================================
# ⚙️ ФУНКЦИИ ЛОГИКИ
# ==========================================

def align_images(img_template, img_pcb):
    """Выравнивание фото платы по шаблону через ORB."""
    gray_tmpl = cv2.cvtColor(img_template, cv2.COLOR_RGB2GRAY)
    gray_pcb = cv2.cvtColor(img_pcb, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_tmpl, None)
    kp2, des2 = orb.detectAndCompute(gray_pcb, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])
    
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    return cv2.warpPerspective(img_pcb, H, (img_template.shape[1], img_template.shape[0]))

def get_binary_mask(image, filter_d, sigma_color, sigma_space, block_size, c_val):
    """Твой пайплайн бинаризации."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, int(filter_d), sigma_color, sigma_space)
    
    bs = int(block_size) if int(block_size) % 2 != 0 else int(block_size) + 1
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, c_val)
    return cv2.bitwise_not(binary)

def inspect_pcb(template, pcb, d, sc, ss, bs, cv, min_area):
    # 1. Выравнивание
    aligned_pcb = align_images(template, pcb)
    
    # 2. Бинаризация обоих
    mask_tmpl = get_binary_mask(template, d, sc, ss, bs, cv)
    mask_pcb = get_binary_mask(aligned_pcb, d, sc, ss, bs, cv)
    
    # 3. Поиск дефектов
    diff_missing = cv2.bitwise_and(mask_tmpl, cv2.bitwise_not(mask_pcb)) # В шаблоне есть, на плате нет
    diff_extra = cv2.bitwise_and(mask_pcb, cv2.bitwise_not(mask_tmpl))   # На плате есть, в шаблоне нет
    
    # Визуализация
    res_img = aligned_pcb.copy()
    for diff, color in [(diff_missing, (255, 0, 0)), (diff_extra, (0, 0, 255))]:
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(res_img, (x, y), (x+w, y+h), color, 2)
                
    return res_img, mask_tmpl, mask_pcb

# ==========================================
# 🖥️ ИНТЕРФЕЙС
# ==========================================
with gr.Blocks(title="PCB AOI System") as demo:
    gr.Markdown("# Система автоматического контроля плат (AOI)")
    with gr.Row():
        with gr.Column():
            img1 = gr.Image(label="Шаблон (Gerber)")
            img2 = gr.Image(label="Фото платы")
            d = gr.Slider(1, 20, 9, label="Bilateral D")
            bs = gr.Slider(3, 99, 9, step=2, label="Block Size")
            c = gr.Slider(-20, 20, 2, label="Constant C")
            min_area = gr.Slider(10, 500, 50, label="Min Area Дефекта")
            btn = gr.Button("Запустить контроль", variant="primary")
            
        with gr.Column():
            out_main = gr.Image(label="Результат дефектоскопии (Red=Extra, Blue=Missing)")
            out_mask1 = gr.Image(label="Маска шаблона")
            out_mask2 = gr.Image(label="Маска платы")

    btn.click(inspect_pcb, inputs=[img1, img2, d, gr.Number(75), gr.Number(75), bs, c, min_area], 
              outputs=[out_main, out_mask1, out_mask2])

demo.launch()