"""人员固定视图。只改变人员目标的展示，不用生成式超分。

默认调用方仍走原来的 thumbnail。本模块在 fixed_view_v1 下：
- 局部图按比例用双三次插值放到固定画幅，编号写在照片上方的标题带；
- 上下文图保留目标周围画面，并用框或外侧角标标出位置；
- 没有人员时，拼图像素与旧 thumbnail 一致。
干净原帧只能由调用方在严格绑定成功后作为 source 传入。本模块不读取 clean_image。
"""
from __future__ import annotations

from PIL import Image, ImageDraw

MAX_PERSON_TARGETS = 16
CELL = 256
HEADER = 24
ROW = CELL + HEADER
COLUMNS = 5
CONTEXT_COLUMNS = 4
LOCAL_LONG_SIDE = CELL
PAD = (114, 114, 114)
INTERPOLATION = 'bicubic'
# 至少向目标两侧各取这些像素，使道路、车辆、建筑仍留在上下文里。
CONTEXT_HALF_MIN = 180


class PersonViewLimit(ValueError):
    """超过固定上限或裁剪为空。调用方应把它当成输入失败，不能当成判假。"""


def is_person(class_name: str) -> bool:
    return class_name in {'人员', '人'}


def legacy_context_box(bbox, width, height):
    """与旧实验 worker 相同：25% 或至少 20 像素，端点交给 Pillow 取整。"""
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    mx, my = max(20, (x2 - x1) * .25), max(20, (y2 - y1) * .25)
    return max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)


def _tight_box(bbox, width, height):
    box = (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
    return _clip(box, width, height)


def _clip(box, width, height):
    x1, y1, x2, y2 = box
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise PersonViewLimit('裁剪范围为空')
    return clipped


def _resample():
    # Pillow 9 只有 Image.BICUBIC；Pillow 9.1 起才有 Image.Resampling。两边都是双三次。
    resampling = getattr(Image, 'Resampling', None)
    if resampling is not None:
        return resampling.BICUBIC
    return Image.BICUBIC


def _fit_long_side(image, long_side):
    width, height = image.size
    if width < 1 or height < 1:
        raise PersonViewLimit('裁剪范围为空')
    scale = long_side / max(width, height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), _resample())
    actual = max(resized.size) / max(width, height)
    plate = Image.new('RGB', (CELL, CELL), PAD)
    offset = ((CELL - resized.width) // 2, (CELL - resized.height) // 2)
    plate.paste(resized, offset)
    return plate, round(actual, 6), list(resized.size), list(offset)


def _context_window(bbox, width, height):
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    half = max(8 * max(x2 - x1, y2 - y1), CONTEXT_HALF_MIN)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return _clip((cx - half, cy - half, cx + half, cy + half), width, height)


def _mark_context(plate, target, crop_box, crop_size, resized_size, offset):
    """只画边或外侧角标，不填充目标内部。"""
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    sx = resized_size[0] / crop_size[0]
    sy = resized_size[1] / crop_size[1]

    def map_x(value):
        return (value - crop_box[0]) / crop_w * crop_size[0] * sx + offset[0]

    def map_y(value):
        return (value - crop_box[1]) / crop_h * crop_size[1] * sy + offset[1]

    rect = [map_x(target[0]), map_y(target[1]), map_x(target[2]), map_y(target[3])]
    draw = ImageDraw.Draw(plate)
    short_side = min(abs(rect[2] - rect[0]), abs(rect[3] - rect[1]))
    if short_side >= 8:
        draw.rectangle(rect, outline='yellow', width=2)
        return 'outline'
    # 目标在上下文里过小，矩形会涂掉它，改在外侧画角标。
    pad = 6
    x1, y1, x2, y2 = [rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad]
    arm = 10
    for corner, dx, dy in (((x1, y1), 1, 1), ((x2, y1), -1, 1), ((x1, y2), 1, -1), ((x2, y2), -1, -1)):
        draw.line([corner, (corner[0] + dx * arm, corner[1])], fill='yellow', width=2)
        draw.line([corner, (corner[0], corner[1] + dy * arm)], fill='yellow', width=2)
    return 'brackets'


def _legacy_cell(source, candidate):
    box = legacy_context_box(candidate['bbox'], source.width, source.height)
    raw = source.crop(box)
    crop = raw.copy()
    crop.thumbnail((CELL, CELL))
    before = max(raw.size)
    scale = 1.0 if max(crop.size) == before else round(max(crop.size) / before, 6)
    return crop, box, scale


def _base_record(candidate, source_kind, presentation):
    return {
        'target_index': candidate['target_index'],
        'class_name': candidate['class_name'],
        'role': presentation,
        'bbox': list(candidate['bbox']),
        'source_kind': source_kind,
        'interpolation': 'thumbnail' if presentation == 'legacy_thumbnail' else INTERPOLATION,
    }


def legacy_sheet(source, candidates, source_kind='screenshot'):
    """旧拼图：只缩小，不放大。供字节对照和非人员单元格复用。"""
    sheet = Image.new('RGB', (COLUMNS * CELL, ((len(candidates) + 4) // COLUMNS) * ROW), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, candidate in enumerate(candidates):
        crop, box, scale = _legacy_cell(source, candidate)
        paste = ((index % COLUMNS) * CELL, (index // COLUMNS) * ROW + HEADER)
        sheet.paste(crop, paste)
        draw.text(((index % COLUMNS) * CELL, (index // COLUMNS) * ROW), str(candidate['target_index']), fill='black')
        tight = _tight_box(candidate['bbox'], source.width, source.height)
        record = _base_record(candidate, source_kind, 'legacy_thumbnail')
        record.update({
            'crop_box': [round(v, 4) for v in box],
            'original_pixel_box': [round(v) for v in tight],
            'original_pixel_size': list(source.crop(tight).size),
            'scale': scale,
            'upscaled': False,
            'paste_xy': list(paste),
            'view_size': list(crop.size),
        })
        records.append(record)
    return sheet, records


def _local_record(source, candidate, source_kind, paste):
    box = _tight_box(candidate['bbox'], source.width, source.height)
    crop = source.crop(box)
    plate, scale, resized, offset = _fit_long_side(crop, LOCAL_LONG_SIDE)
    record = _base_record(candidate, source_kind, 'fixed_local')
    record.update({
        'crop_box': [round(v, 4) for v in box],
        'original_pixel_box': [round(v) for v in box],
        'original_pixel_size': list(crop.size),
        'scale': scale,
        'upscaled': scale > 1,
        'paste_xy': list(paste),
        'view_size': [CELL, CELL],
        'resized_size': resized,
        'letterbox_xy': offset,
        'pad_rgb': list(PAD),
    })
    return plate, record


def _context_record(source, candidate, source_kind, paste, legacy_height):
    window = _context_window(candidate['bbox'], source.width, source.height)
    target = _tight_box(candidate['bbox'], source.width, source.height)
    crop = source.crop(window)
    plate, scale, resized, offset = _fit_long_side(crop, LOCAL_LONG_SIDE)
    marker = _mark_context(plate, target, window, crop.size, tuple(resized), tuple(offset))
    record = _base_record(candidate, source_kind, 'fixed_context')
    record.update({
        'crop_box': [round(v, 4) for v in window],
        'original_pixel_box': [round(v) for v in target],
        'original_pixel_size': list(source.crop(target).size),
        'scale': scale,
        'upscaled': scale > 1,
        'paste_xy': [paste[0], legacy_height + paste[1] + HEADER],
        'view_size': [CELL, CELL],
        'marker': marker,
        'pad_rgb': list(PAD),
    })
    return plate, record


class RenderedSheet:
    def __init__(self, sheet, records, changed, source_kind):
        self.sheet = sheet
        self.records = records
        self.changed = changed
        self.source_kind = source_kind


def render_sheet(source, candidates, source_kind='screenshot'):
    """有人员时替换人员单元格并在下方追加上下文；没有人员时与旧拼图一致。"""
    if source_kind not in {'screenshot', 'bound_clean_frame'}:
        raise PersonViewLimit('未知图片来源')
    persons = [c for c in candidates if is_person(c['class_name'])]
    if len(persons) > MAX_PERSON_TARGETS:
        raise PersonViewLimit(f'人员目标 {len(persons)} 个，超过每张图 {MAX_PERSON_TARGETS} 个的上限')
    if not persons:
        sheet, records = legacy_sheet(source, candidates, source_kind)
        return RenderedSheet(sheet, records, False, source_kind)
    legacy_height = ((len(candidates) + 4) // COLUMNS) * ROW
    context_height = ((len(persons) + CONTEXT_COLUMNS - 1) // CONTEXT_COLUMNS) * ROW
    sheet = Image.new('RGB', (COLUMNS * CELL, legacy_height + context_height), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, candidate in enumerate(candidates):
        origin = ((index % COLUMNS) * CELL, (index // COLUMNS) * ROW)
        if is_person(candidate['class_name']):
            plate, record = _local_record(source, candidate, source_kind, (origin[0], origin[1] + HEADER))
            sheet.paste(plate, record['paste_xy'])
        else:
            crop, box, scale = _legacy_cell(source, candidate)
            paste = (origin[0], origin[1] + HEADER)
            sheet.paste(crop, paste)
            tight = _tight_box(candidate['bbox'], source.width, source.height)
            record = _base_record(candidate, source_kind, 'legacy_thumbnail')
            record.update({
                'crop_box': [round(v, 4) for v in box],
                'original_pixel_box': [round(v) for v in tight],
                'original_pixel_size': list(source.crop(tight).size),
                'scale': scale,
                'upscaled': False,
                'paste_xy': list(paste),
                'view_size': list(crop.size),
            })
        draw.text(origin, str(candidate['target_index']), fill='black')
        records.append(record)
    for index, candidate in enumerate(persons):
        origin = ((index % CONTEXT_COLUMNS) * CELL, (index // CONTEXT_COLUMNS) * ROW)
        plate, record = _context_record(source, candidate, source_kind, origin, legacy_height)
        sheet.paste(plate, (origin[0], legacy_height + origin[1] + HEADER))
        draw.text((origin[0], legacy_height + origin[1]), str(candidate['target_index']), fill='black')
        records.append(record)
    return RenderedSheet(sheet, records, True, source_kind)
