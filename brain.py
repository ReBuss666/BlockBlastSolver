import cv2
import numpy as np
import os

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ИЗВЕСТНЫЕ ФИГУРЫ Block Blast (включая диагональные полоски) ---
KNOWN_SHAPES = [
    # Одиночный блок
    ("1x1",       np.array([[1]])),
    # Линии
    ("line_1x2",  np.array([[1, 1]])),
    ("line_2x1",  np.array([[1], [1]])),
    ("line_1x3",  np.array([[1, 1, 1]])),
    ("line_3x1",  np.array([[1], [1], [1]])),
    ("line_1x4",  np.array([[1, 1, 1, 1]])),
    ("line_4x1",  np.array([[1], [1], [1], [1]])),
    ("line_1x5",  np.array([[1, 1, 1, 1, 1]])),
    ("line_5x1",  np.array([[1], [1], [1], [1], [1]])),
    # Квадраты
    ("sq_2x2",    np.array([[1, 1], [1, 1]])),
    ("sq_3x3",    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
    # L-фигуры (2x2)
    ("L2",        np.array([[1, 0], [1, 1]])),
    ("L2",        np.array([[1, 1], [1, 0]])),
    ("L2",        np.array([[1, 1], [0, 1]])),
    ("L2",        np.array([[0, 1], [1, 1]])),
    # L-фигуры (3x3)
    ("L3",        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])),
    ("L3",        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])),
    ("L3",        np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])),
    ("L3",        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])),
    # T-фигуры
    ("T",         np.array([[1, 1, 1], [0, 1, 0]])),
    ("T",         np.array([[0, 1, 0], [1, 1, 1]])),
    ("T",         np.array([[1, 0], [1, 1], [1, 0]])),
    ("T",         np.array([[0, 1], [1, 1], [0, 1]])),
    # S/Z фигуры
    ("S",         np.array([[0, 1, 1], [1, 1, 0]])),
    ("S",         np.array([[1, 1, 0], [0, 1, 1]])),
    ("S",         np.array([[1, 0], [1, 1], [0, 1]])),
    ("S",         np.array([[0, 1], [1, 1], [1, 0]])),
    # === Диагональные полоски 2x2 ===
    ("diag2",     np.array([[1, 0], [0, 1]])),
    ("diag2",     np.array([[0, 1], [1, 0]])),
    # === Диагональные полоски 3x3 ===
    ("diag3",     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    ("diag3",     np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),
]

# --- ЗРЕНИЕ ---

def get_slot_roi(img, slot_center, slot_size):
    """Вырезает квадратную область вокруг центра слота."""
    cx, cy = slot_center
    half = slot_size // 2
    y1, y2 = max(0, cy - half), min(img.shape[0], cy + half)
    x1, x2 = max(0, cx - half), min(img.shape[1], cx + half)
    return img[y1:y2, x1:x2]


def _trim_shape(matrix):
    """Убирает нулевые строки и столбцы по краям матрицы фигуры."""
    arr = np.array(matrix)
    if arr.size == 0 or np.sum(arr) == 0:
        return None
    rows_mask = np.any(arr == 1, axis=1)
    arr = arr[rows_mask]
    cols_mask = np.any(arr == 1, axis=0)
    arr = arr[:, cols_mask]
    return arr


def _match_known_shape(matrix):
    """Сопоставляет распознанную матрицу с известными фигурами."""
    trimmed = _trim_shape(matrix)
    if trimmed is None:
        return None, None

    best_name = None
    best_shape = None
    best_diff = 999

    for name, tmpl in KNOWN_SHAPES:
        if tmpl.shape == trimmed.shape:
            diff = int(np.sum(np.abs(trimmed - tmpl)))
            if diff < best_diff:
                best_diff = diff
                best_name = name
                best_shape = tmpl

    # Допускаем до 1 ошибочной клетки (шум при распознавании)
    if best_diff <= 1:
        return best_name, best_shape

    return None, None


def recognize_slot(full_img, slot_center, board_rect, slot_index=0):
    """
    Распознаёт фигуру в слоте.
    Использует HSV-детекцию цветных блоков + Otsu как запасной метод.
    """
    bx1, by1, bx2, by2 = board_rect
    board_cell_size = (bx2 - bx1) / 8.0
    slot_block_size = board_cell_size * 0.55  # масштаб фигур в слотах

    # ROI: не больше ширины одного слота (~board_w/3), чтобы не захватить соседей
    roi_size = int(board_cell_size * 2.8)
    roi = get_slot_roi(full_img, slot_center, roi_size)
    if roi.size == 0:
        return "Error", None

    # --- Метод 1: HSV — ищем яркие цветные блоки ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    # Цветные блоки = высокая насыщенность И яркость
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    mask[(s_ch > 40) & (v_ch > 100)] = 255

    # Если HSV нашёл слишком мало пикселей — пробуем Otsu
    fill = np.sum(mask > 0) / mask.size
    if fill < 0.03:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Морфология: убираем шум, соединяем близкие блоки
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Отладка: сохраняем ROI и маску
    cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_slot{slot_index}_roi.png"), roi)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_slot{slot_index}_mask.png"), mask)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Empty", None

    # Фильтруем мелкие контуры (шум)
    min_area = slot_block_size * slot_block_size * 0.3
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        return "Empty", None

    # Объединяем валидные контуры → общий bounding rect
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Строим заполненную маску только из валидных контуров
    combined_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(combined_mask, valid_contours, -1, 255, -1)
    shape_roi = combined_mask[y:y+h, x:x+w]

    # Определяем размер сетки
    cols = int(round(w / slot_block_size))
    rows = int(round(h / slot_block_size))
    cols, rows = max(1, min(5, cols)), max(1, min(5, rows))

    # Собираем матрицу по заполненности центра каждой ячейки
    matrix = []
    cw, ch_cell = w / cols, h / rows

    for r in range(rows):
        row_data = []
        for c in range(cols):
            cy1 = int((r + 0.25) * ch_cell)
            cy2 = int((r + 0.75) * ch_cell)
            cx1 = int((c + 0.25) * cw)
            cx2 = int((c + 0.75) * cw)
            cy1, cy2 = max(0, cy1), min(h, cy2)
            cx1, cx2 = max(0, cx1), min(w, cx2)
            zone = shape_roi[cy1:cy2, cx1:cx2]
            fill_ratio = np.mean(zone) / 255.0 if zone.size > 0 else 0
            row_data.append(1 if fill_ratio > 0.3 else 0)
        matrix.append(row_data)

    raw_matrix = np.array(matrix)

    # Отладка: сохраняем вырез фигуры
    cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_slot{slot_index}_shape.png"), shape_roi)

    # Обрезаем нулевые края
    trimmed = _trim_shape(raw_matrix)
    if trimmed is None or np.sum(trimmed) == 0:
        return "Empty", None

    # Пытаемся сопоставить с известной фигурой
    known_name, known_shape = _match_known_shape(trimmed)
    if known_shape is not None:
        return known_name, known_shape

    # Не нашли точного совпадения — возвращаем как есть
    sh, sw = trimmed.shape
    return f"{sh}x{sw}", trimmed

# --- ИНТЕЛЛЕКТ (ЛОГИКА ИГРЫ) ---

def can_place(board, shape, r, c):
    sh, sw = shape.shape
    if r + sh > 8 or c + sw > 8: return False
    for i in range(sh):
        for j in range(sw):
            if shape[i][j] == 1 and board[r+i][c+j] == 1:
                return False
    return True

def simulate_move(board, shape, r, c):
    new_board = board.copy()
    sh, sw = shape.shape
    for i in range(sh):
        for j in range(sw):
            if shape[i][j] == 1: new_board[r+i][c+j] = 1
    
    # Считаем очищенные линии
    lines = 0
    full_rows = np.all(new_board == 1, axis=1)
    full_cols = np.all(new_board == 1, axis=0)
    lines = np.sum(full_rows) + np.sum(full_cols)
    
    # Очищаем их в симуляции
    new_board[full_rows, :] = 0
    new_board[:, full_cols] = 0
    return new_board, lines

def evaluate_board(board):
    """Оценка надежности позиции (чем выше, тем лучше)."""
    # Штраф за занятые клетки
    occupied = np.sum(board)
    # Штраф за "дырки" (пустые клетки, окруженные занятыми)
    holes = 0
    for r in range(1, 7):
        for c in range(1, 7):
            if board[r][c] == 0:
                if board[r-1][c] and board[r+1][c] and board[r][c-1] and board[r][c+1]:
                    holes += 1
    return -(occupied * 1 + holes * 20)

def solve_step(board, available_shapes):
    """Ищет лучшую последовательность для всех доступных фигур."""
    best_sequence = []
    max_total_score = -999999

    # Для простоты и скорости: ищем лучший ход для каждой фигуры по очереди,
    # учитывая изменения на доске после каждой поставленной фигуры.
    current_board = board.copy()
    final_moves = []
    
    # Сортируем фигуры от сложных (больших) к легким
    shapes_to_process = sorted(available_shapes, key=lambda x: np.sum(x[2]), reverse=True)

    for idx, name, shape in shapes_to_process:
        best_move = None
        best_move_score = -999999
        
        for r in range(8):
            for c in range(8):
                if can_place(current_board, shape, r, c):
                    temp_board, cleared = simulate_move(current_board, shape, r, c)
                    score = evaluate_board(temp_board) + (cleared * 500)
                    
                    if score > best_move_score:
                        best_move_score = score
                        best_move = (idx, r, c, temp_board)
        
        if best_move:
            idx, r, c, next_board = best_move
            final_moves.append((idx, r, c))
            current_board = next_board
        else:
            print(f"!!! Не нашел места для {name}")

    return final_moves

def print_shape(label, shape):
    print(f"{label}:")
    for row in shape: print("".join(["[]" if x else "  " for x in row]))