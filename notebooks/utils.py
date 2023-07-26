from collections import defaultdict
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# from config import CONFIG_NGRAMS_NULL
# from generate_samples.utils import crop_image_from_bboxes
from utils import load_rgb

DPI = 10


def show(
    image,
    ax=None,
    rotate=False,
    scale=1,
    title=None,
    fontsize=10,
    cmap=None,
    return_ax=False,
):
    h, w = image.shape[:2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(int(scale * h), int(scale * w)), dpi=DPI)
    ax.cla()
    ax.axis("off")
    if rotate:
        param = (1, 0) if image.ndim == 2 else (1, 0, 2)
        image = np.transpose(image, param)[::-1]
    ax.imshow(image, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=DPI * fontsize)
    if return_ax:
        return ax


def show_collection(
    images,
    titles=[],
    num_rows=-1,
    num_cols=-1,
    scale=1,
    cmap=None,
    return_axes=False,
    fontsize=10,
    # pad=1.0,
):
    assert len(images) > 1

    if num_cols == -1:
        # none provided: fix row in 1
        if num_rows == -1:
            num_rows = 1
        # compute #cols based on #rows
        num_cols = len(images) // num_rows
    else:
        # only #cols provided: fix cols, compute rows
        if num_rows == -1:
            num_rows = math.ceil(len(images) / num_cols)
        # both rows and cols provided: fix rows, compute cols
        else:
            num_cols = math.ceil(len(images) / num_rows)

    if len(titles) > 0:
        assert len(titles) == len(images)
    else:
        titles = len(images) * [""]

    h, w = images[0].shape[:-1]
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(int(scale * w * num_cols), int(scale * h * num_rows)),
        dpi=DPI,
    )
    k = 1
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image, cmap=cmap)
        ax.axis("off")
        ax.set_title(title, fontsize=DPI * fontsize)
        k += 1
    fig.tight_layout()
    if return_axes:
        return axes


def draw_bbox(ax, bbox, color=(1.0, 0, 1.0)):
    rect = Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor=color, facecolor="none"
    )
    ax.add_artist(rect)
    return ax


def gray2rgb(image):
    return np.transpose(np.stack(3 * [image]), (1, 2, 0))


def list2table(mlist, n=20):
    line = []
    lines = [" | ".join(n * ["[]()"]), "|".join(n * ["-----"])]
    k = 0
    for w in mlist:
        line.append(w)
        if (k + 1) % n == 0:
            lines.append(" | ".join(line))
            line = []
        k += 1
    sep = "\n{}\n".format("|".join(n * ["-----"]))
    text = "\n".join(lines)
    return text


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height.
    https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{:.2f}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


# def show_samples(
#     ngrams_dict,
#     k=100,
#     filter_negative=False,
#     num_rows=10,
#     scale=0.5,
#     fontsize=12,
#     pad=1.0,
#     sample_size=(32, 64),
#     shuffle=False,
# ):
#     # filter
#     samples_all = []
#     for ngram_text, samples in ngrams_dict.items():
#         negative = ngram_text == CONFIG_NGRAM_NEGATIVE_CHAR
#         unused = ngram_text == CONFIG_NGRAM_UNUSED_CHAR
#         used = not (negative or unused)
#         # condition_positive = filter_positive and positive
#         # print(condition_unused, len(samples))
#         if filter_used and used:
#             continue
#         if filter_unused and unused:
#             continue
#         if filter_negative and negative:
#             continue
#         for doc1, doc2, bbox1, bbox2, conf, mode in samples:
#             sample_with_ngram = (ngram_text, doc1, doc2, bbox1, bbox2, conf, mode)
#             samples_all.append(sample_with_ngram)

#     # show
#     samples_img = []
#     titles = []
#     instances = random.choices(samples_all, k=k)
#     for ngram_text, doc1, doc2, bbox1, bbox2, _, _ in instances:
#         # pick a sample
#         # ngram_text, doc1, doc2, bbox1, bbox2 = random.choice(samples_all)
#         image1 = load_rgb(doc1.replace("_ori", "_bin"))
#         image2 = load_rgb(doc2.replace("_ori", "_bin"))
#         sample_img = crop_image_from_bboxes(image1, image2, bbox1, bbox2)
#         samples_img.append(put_frame(sample_img))
#         titles.append(ngram_text)

#     show_collection(samples_img, num_rows=num_rows, titles=titles, scale=scale, fontsize=fontsize)


# def plot_ngram_dist(ngrams_dict, top_ngrams=30):
#     fig, ax = plt.subplots(figsize=(15, 3))
#     count = {ngram_text: len(samples) for ngram_text, samples in ngrams_dict.items()}
#     count_sorted = sorted(count.items(), key=lambda item: item[1], reverse=True)[:top_ngrams]
#     labels, values = zip(*count_sorted)
#     # labels
#     plt.bar(range(len(count_sorted)), values, align="center")
#     plt.xticks(range(len(count_sorted)), labels, fontsize=12)
#     plt.yticks(fontsize=12)


def plot_dist(values, topn=30):
    fig, ax = plt.subplots(figsize=(15, 3))
    count = defaultdict(lambda: 0)
    for val in values:
        count[val] += 1
    count_sorted = sorted(count.items(), key=lambda item: item[1], reverse=True)[:topn]
    labels, values = zip(*count_sorted)
    # labels
    plt.bar(range(len(count_sorted)), values, align="center")
    plt.xticks(range(len(count_sorted)), labels, fontsize=12)
    plt.yticks(fontsize=12)


def put_frame(image, is_rgb=False, thickness=1, color=(255, 0, 0)):
    h, w = image.shape[:2]
    bbox = (0, 0, w, h)
    image_frame = draw_bbox(image, bbox, is_rgb, thickness, color=color)
    return image_frame


def draw_bbox(image, bbox, is_rgb=False, thickness=1, color=(255, 0, 0)):
    if image.dtype in [np.bool, np.float32]:
        image = (255 * image).astype(np.uint8)
    if not is_rgb:
        image = np.stack([image, image, image], axis=0).transpose([1, 2, 0])

    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    image = image.copy()
    image_rec = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image_rec


# def sample_with_frame(sample, is_rgb=False, thickness=1, color=(255, 0, 0)):
#     if not is_rgb:
#         # sample_bin = (255 * sample)
#         sample = np.stack([sample, sample, sample], axis=0).transpose([1, 2, 0])
#     sample = (255 * sample).astype(np.uint8)
#     return put_frame(sample, thickness, color)


def replace_underscore(df):
    map_columns = {column: column.replace("_", "-") for column in df.columns}
    df.rename(map_columns, axis="columns", inplace=True)
    # check which columns are strings
    str_columns = df.applymap(type).eq(str).all()
    for column in df.columns:
        if str_columns[column]:
            map_values = {
                value: value.replace("_", "-")
                for value in df[column].unique()
                if type(value) == str
            }
            df[column].replace(map_values, inplace=True)