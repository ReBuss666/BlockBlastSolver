import subprocess
import time
import re
import cv2
import numpy as np
import os

# =====================================================================
# Путь к adb.exe  (raw-строка r"..." чтобы обратные слеши не ломались)
# Укажи свой актуальный путь!
# =====================================================================
ADB_CMD = r"D:\Programming\CppVS\VS Code\projects\cheat_for_blockblast\adb.exe"

# Смещение при перетаскивании (чтобы палец/курсор не закрывал фигуру)
OFFSET = 600

# Папка для сохранения отладочных скриншотов (текущая директория)
DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  ADB Функции
# =====================================================================

def adb_run(args):
    """Запускает adb с аргументами и возвращает stdout (bytes)."""
    cmd = [ADB_CMD] + args
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{ADB_CMD}' не найден.\n"
            "Скачай platform-tools и положи adb.exe рядом со скриптом, "
            "или укажи правильный путь в ADB_CMD."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ADB вернул ошибку:\n{e.output.decode('utf-8', errors='replace')}\n"
            "Проверь подключение устройства и USB-отладку."
        )

def adb_get_device_size():
    """Возвращает (width, height) физического экрана устройства."""
    out = adb_run(["shell", "wm", "size"]).decode("utf-8")
    m = re.search(r'(\d+)x(\d+)', out)
    if not m:
        raise RuntimeError(f"Не удалось распарсить размер экрана: {out}")
    return int(m.group(1)), int(m.group(2))

def adb_screenshot():
    """Делает скриншот устройства через ADB и возвращает cv2-изображение (BGR)."""
    png_data = adb_run(["exec-out", "screencap", "-p"])
    arr = np.frombuffer(png_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не удалось декодировать скриншот с устройства.")
    return img

def adb_swipe(x1, y1, x2, y2, duration_ms=400):
    """Свайп (drag) на устройстве: зажать (x1,y1) → протянуть к (x2,y2)."""
    adb_run(["shell", "input", "swipe",
             str(int(x1)), str(int(y1)),
             str(int(x2)), str(int(y2)),
             str(int(duration_ms))])

def adb_tap(x, y):
    """Тап по координатам на устройстве."""
    adb_run(["shell", "input", "tap", str(int(x)), str(int(y))])

# =====================================================================
#  Автоматический поиск игрового поля
# =====================================================================

def _cluster_lines(values, min_gap):
    """Кластеризует отсортированные координаты линий."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.mean(c)) for c in clusters]

def auto_detect_board(img):
    """
    Автоматически находит игровое поле 8x8 на скриншоте Block Blast.
    Возвращает (x1, y1, x2, y2) — прямоугольник поля в пикселях устройства.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # --- Метод 1: поиск крупнейшего квадратного контура ---
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.08:      # меньше 8% экрана — слишком мало
            continue
        if area > w * h * 0.75:      # больше 75% — это весь экран
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.7 < aspect < 1.4 and area > best_area:
            best = (x, y, x + bw, y + bh)
            best_area = area

    if best:
        print(f"  [detect] Найдено поле (по контурам), area={best_area}")
        return best

    # --- Метод 2: поиск по цвету фона (резервный) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 15), (180, 120, 90))
    kernel2 = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=2)

    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best2 = None
    best_area2 = 0
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area < w * h * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.7 < aspect < 1.4 and area > best_area2:
            best2 = (x, y, x + bw, y + bh)
            best_area2 = area

    if best2:
        print(f"  [detect] Найдено поле (по цвету фона)")
        return best2

    return None

def auto_detect_slots(img, board_rect):
    """Определяет положение 3 слотов ниже игрового поля."""
    bx1, by1, bx2, by2 = board_rect
    h, w = img.shape[:2]
    board_w = bx2 - bx1
    board_cx = (bx1 + bx2) // 2

    # Слоты обычно на 40-50% расстояния от низа поля до низа экрана
    gap = h - by2
    slot_y = by2 + int(gap * 0.45)

    # 3 слота равномерно по ширине поля
    slot_spacing = board_w // 3
    slot1_x = board_cx - slot_spacing
    slot2_x = board_cx
    slot3_x = board_cx + slot_spacing

    return [(slot1_x, slot_y), (slot2_x, slot_y), (slot3_x, slot_y)]

# =====================================================================
#  Логика игрового поля
# =====================================================================

def get_cell_center(board_rect, row, col):
    """Возвращает (x, y) центра клетки [row][col]."""
    bx1, by1, bx2, by2 = board_rect
    cell_w = (bx2 - bx1) / 8
    cell_h = (by2 - by1) / 8
    cx = bx1 + col * cell_w + cell_w / 2
    cy = by1 + row * cell_h + cell_h / 2
    return int(cx), int(cy)

def get_board_state(img, board_rect):
    """Анализирует поле 8x8 и возвращает матрицу (0 = пусто, 1 = занято)."""
    bx1, by1, bx2, by2 = board_rect
    board_img = img[by1:by2, bx1:bx2]
    bh, bw, _ = board_img.shape
    cell_h = bh // 8
    cell_w = bw // 8

    R = max(3, min(cell_h, cell_w) // 6)
    cell_colors = np.zeros((8, 8, 3), dtype=float)
    
    for row in range(8):
        for col in range(8):
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2
            ry1 = max(cy - R, 0)
            ry2 = min(cy + R, bh)
            rx1 = max(cx - R, 0)
            rx2 = min(cx + R, bw)
            region = board_img[ry1:ry2, rx1:rx2]
            cell_colors[row, col] = region.mean(axis=(0, 1))

    # Эталон пустой клетки = медиана всех 64 цветов
    all_colors = cell_colors.reshape(64, 3)
    bg_color = np.median(all_colors, axis=0)

    DIFF_THRESHOLD = 35  # порог отличия от фона

    matrix = [[0] * 8 for _ in range(8)]
    debug_img = img.copy()

    for row in range(8):
        for col in range(8):
            diff = np.sqrt(np.sum((cell_colors[row, col] - bg_color) ** 2))
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2

            if diff > DIFF_THRESHOLD:
                matrix[row][col] = 1
                cv2.circle(debug_img, (bx1 + cx, by1 + cy), 12, (0, 0, 255), -1)
            else:
                cv2.circle(debug_img, (bx1 + cx, by1 + cy), 6, (0, 255, 0), 1)

    return debug_img, matrix

def move_piece(board_rect, slot_x, slot_y, target_row, target_col, offset=OFFSET):
    """Перетаскивает фигуру из слота в нужную клетку поля."""
    tx, ty = get_cell_center(board_rect, target_row, target_col)
    ty_corrected = ty + offset
    print(f"  swipe ({slot_x},{slot_y}) -> [{target_row}][{target_col}] = ({tx},{ty_corrected})")
    adb_swipe(slot_x, slot_y, tx, ty_corrected, duration_ms=400)
    time.sleep(0.3)

# =====================================================================
#  ЗАПУСК
# =====================================================================

if __name__ == '__main__':
    print("=== Block Blast ADB Player (Contour Edition) ===\n")

    print("[1] Проверяю ADB...")
    dev_w, dev_h = adb_get_device_size()
    print(f"    Устройство: {dev_w}x{dev_h}")

    print("[2] Делаю скриншот...")
    screenshot = adb_screenshot()
    
    raw_path = os.path.join(DEBUG_DIR, "debug_raw.png")
    cv2.imwrite(raw_path, screenshot)
    print(f"    Сохранён: {raw_path}")

    print("[3] Ищу игровое поле...")
    board_rect = auto_detect_board(screenshot)

    if board_rect is None:
        print("    НЕ УДАЛОСЬ найти поле автоматически!")
        raise SystemExit(1)

    bx1, by1, bx2, by2 = board_rect
    print(f"    Поле найдено: ({bx1}, {by1}) - ({bx2}, {by2}) размер {bx2-bx1}x{by2-by1}")

    slots = auto_detect_slots(screenshot, board_rect)
    print(f"    Слоты: {slots}")

    print("[4] Анализирую клетки поля...")
    debug_img, board = get_board_state(screenshot, board_rect)

    print("\n--- СОСТОЯНИЕ ПОЛЯ ---")
    for r in board:
        print(" ".join(["■" if c else "·" for c in r]))

    # Импортируем мозг для контурного распознавания фигур
    from brain import recognize_slot, solve_step, print_shape

    print("\n[5] Распознаю фигуры в слотах по контуру...")
    available_shapes = []
    
    for i, (sx, sy) in enumerate(slots):
        name, shape = recognize_slot(screenshot, (sx, sy), board_rect, slot_index=i)
        
        # Отрисовка слотов для дебага
        cv2.circle(debug_img, (sx, sy), 20, (0, 255, 255), 3)
        cv2.putText(debug_img, str(i+1), (sx-8, sy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if shape is not None and name != "Empty" and name != "Error":
            print_shape(f"  Слот {i+1}: {name}", shape)
            available_shapes.append((i, name, shape))
        else:
            print(f"  Слот {i+1}: пуст или не распознан")

    # Сохраняем и показываем отладочную картинку
    debug_path = os.path.join(DEBUG_DIR, "debug_detected.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"\nОтладочная картинка сохранена: {debug_path}")

    if not available_shapes:
        print("\nНет фигур для размещения (или они не распознаны).")
        raise SystemExit(0)

    board_np = np.array(board, dtype=int)
    
    print("\n[6] Ищу лучшие ходы...")
    moves = solve_step(board_np, available_shapes)

    if not moves:
        print("Не смог найти ни одного хода! Возможно, нет места.")
        raise SystemExit(0)

    print(f"\nНайдено {len(moves)} ходов:")
    for slot_idx, r, c in moves:
        name = available_shapes[slot_idx][1] if slot_idx < len(available_shapes) else "?"
        print(f"  Слот {slot_idx+1} ({name}) -> [{r}][{c}]")

    answer = input("\nВыполнить ходы на устройстве? (y/n): ").strip().lower()
    if answer == 'y':
        for slot_idx, r, c in moves:
            # Ищем оригинальный индекс слота (т.к. некоторые слоты могли быть пустыми)
            real_slot_index = available_shapes[slot_idx][0]
            sx, sy = slots[real_slot_index]
            
            print(f"  Ход: слот {real_slot_index+1} -> [{r}][{c}]...")
            move_piece(board_rect, sx, sy, r, c)
            time.sleep(0.5) # Пауза между свайпами, чтобы анимация в игре успела пройти
        print("\nГотово! Все ходы выполнены.")
    else:
        print("Отменено.")