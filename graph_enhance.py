from load_data import load_data
from parameter import parse_args
import numpy as np
import copy
import re
import os
def correct_sent(sent, event_word):
    assert isinstance(sent, str) and isinstance(event_word, str)
    event_word_num = len(event_word.split(" "))

    pattern = re.compile(r"(?<![\w-])(" + re.escape(event_word) + r")(?![\w-])")
    match = pattern.search(sent)

    if not match:
        pattern_ci = re.compile(r"(?<![\w-])(" + re.escape(event_word) + r")(?![\w-])", re.IGNORECASE)
        match = pattern_ci.search(sent)

    if not match:
        if event_word not in sent:
            print(sent)
            print(repr(event_word))
            raise AssertionError("event_word not found in sentence")
        start = sent.find(event_word)
        end = start + len(event_word)
    else:
        start, end = match.span(1)

    marked = sent[:start] + "<EVT>" + sent[end:]
    marked = re.sub(r'([,.;:!?\"“”‘’()\[\]{}])', r' \1 ', marked)
    marked = re.sub(r'\s*<EVT>\s*', ' <EVT> ', marked)
    new_sent = re.sub(r"\s+", " ", marked).strip()
    new_sent = new_sent.replace('<EVT>', event_word)
    new_sent = re.sub(r"\s+", " ", new_sent).strip()

    tmp = pattern.sub("<EVT>", new_sent, count=1)
    words = tmp.split(" ")
    try:
        position = words.index("<EVT>")
    except ValueError:
        tmp_ci = re.sub(r"(?i)(?<![\w-])(" + re.escape(event_word) + r")(?![\w-])", "<EVT>", new_sent, count=1)
        words = tmp_ci.split(" ")
        position = words.index("<EVT>")

    pos_word = "_" + "_".join(str(p) for p in range(position, position + event_word_num))
    return new_sent, pos_word
# 加载参数
args = parse_args()
args.graph_enhance = "None"
data_path = args.dataset_path
graph_data_path =os.path.join(data_path, "enhance_graph")
train_data, dev_data, test_data = load_data(args)
data = {"train": train_data, "dev": dev_data, "test": test_data}
train_size, dev_size, test_size = len(train_data), len(dev_data), len(test_data)
print('Data loaded')

train_events_result = np.load(os.path.join(graph_data_path, "enhance_train_events_result.npy"), allow_pickle=True).item()
dev_events_result = np.load(os.path.join(graph_data_path, "enhance_dev_events_result.npy"), allow_pickle=True).item()
test_events_result = np.load(os.path.join(graph_data_path, "enhance_test_events_result.npy") , allow_pickle=True).item()
events_result = {"train" : train_events_result, "dev": dev_events_result, "test": test_events_result}

# 将补丁添加进原有数据集
for k in ["train", "dev", "test"]:
    global_event_idx = 0

    for i, sample in enumerate(data[k]):
        num_nodes = len(sample['node'])

        gen_events = []
        gen_edges = []

        for j, edge in enumerate(sample['edge'][:-1]):
            if global_event_idx < len(events_result[k]['all_events_C_desc']):
                event_result = events_result[k]['all_events_C_desc'][global_event_idx]
                if event_result[0] and event_result[1] and \
                        event_result[0] != "" and event_result[1] != "" and \
                        event_result[1] != "not_in" and event_result[1] != "format":
                    corrected_sent, pos = correct_sent(event_result[1], event_result[0])
                    gen_events.append((-1, '-1', -1, -1, '-1', event_result[0], corrected_sent, -1, pos))

                    gen_event_id = num_nodes + len(gen_events) - 1
                    gen_edge_1 = [edge[0], 'cause', gen_event_id]
                    gen_edge_2 = [gen_event_id, 'cause', edge[-1]]
                    gen_edges.append(gen_edge_1)
                    gen_edges.append(gen_edge_2)
            global_event_idx += 1
        data[k][i]['node'].extend(gen_events)
        data[k][i]['edge'] = data[k][i]['edge'][:-1] + gen_edges + [data[k][i]['edge'][-1]]
np.save(os.path.join(data_path, 'MAVENSubWoRe_en.npy'), {'train': data['train'], 'valid': data['dev'], 'test': data['test']})
