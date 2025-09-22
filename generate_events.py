# -*- coding: utf-8 -*-
import transformers
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from load_data import load_data
from parameter import parse_args
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import numpy as np
from tqdm import tqdm
import json

assessment_template = """- **Role**: Causal Relationship Assessment Expert  
- **Background**: The user needs to evaluate a given pair of causal events to determine whether the causal relationship between them is sufficiently clear. This assessment serves as the basis for deciding whether to generate intermediate events later, ensuring the completeness and logical coherence of the causal graph.  
- **Profile**: You are an expert in causal relationship assessment, with a profound background in logic and semantic analysis. You can accurately judge the logical relationship between pairs of causal events and determine whether it is necessary to add intermediate events to enhance the rationality of the causal chain.  
- **Skills**: You are skilled in causal logic analysis and can accurately determine whether a pair of causal events requires further supplementation with intermediate events.  
- **Goals**: Evaluate the given pair of causal events, analyze their causal relationship, and provide a clear conclusion on whether it is necessary to generate intermediate events.  
- **Constrains**: The assessment should be based on causal logic and semantic consistency to ensure the accuracy and reliability of the judgment. The output should clearly indicate whether it is necessary to generate intermediate events without generating specific intermediate events.  
- **OutputFormat**: The output should be directly presented in JSON format, containing the following two fields:
  - `\"analysis\"`: A string that provides a detailed description of the logical relationship between the cause-and-effect event pair, including the strength of the causal chain and whether there are any logical loopholes.
  - `\"need_intermediate_event\"`: A boolean value indicating whether an intermediate event needs to be generated.
- **Input**:  
    - Event word for the cause event: \"%s\"  
    - Original description sentence for the cause event: \"%s\"  
    - Event word for the effect event: \"%s\"  
    - Original description sentence for the effect event: \"%s\""""

generate_template = """- **Role**: Causal Graph Topology Enhancement Expert
- **Background**: The user requires a tool capable of generating rational intermediary events to enhance the topology of causal graphs. This is primarily aimed at improving the performance and robustness of causal graph event prediction models when dealing with incomplete data and cross-scenario reasoning. It addresses the issue of performance degradation in existing models when encountering event labels not present in the training set, as well as the problem of missing key event clues during the data collection process.
- **Profile**: You are an expert proficient in causal graph topology enhancement, with an in-depth understanding and modeling of causal relationships. You are skilled at leveraging large language models to generate intermediary events that satisfy causal propagation logic, especially when referencing prior analysis of the insufficiency in causal relationships between event pairs.
- **Skills**: You possess professional skills in causal graph modeling and the application of large language models. You are able to accurately generate intermediary events that conform to causal logic, with the generation process referring to the analysis of the situation where the causal relationship of the given causal event pair is not significant in the previous step, and naturally insert them into existing causal chains.
- **Goals**:
  1. With reference to the analysis of the insufficiently significant causal relationship between the given causal event pair in the previous step, generate a rational intermediary event based on the given causal event pair, ensuring it conforms to causal propagation logic and bridges the gap identified in the analysis.
  2. Ensure that the generated intermediary event can be naturally inserted into the existing causal chain, forming the structure "Event A → Intermediary Event C → Event B," and that this insertion effectively addresses the issues pointed out in the prior analysis of the weak causal relationship.
  3. Provide the original descriptive sentence containing the intermediary event C, maintaining a neutral and objective sentence style, which should align with the context implied by the previous analysis.
- **Constraints**: The generated intermediary event must conform to causal logic and be able to be naturally inserted into the existing causal chain.
- **Output Format**:
  - `\"Event C Event Word\"`: A string that represents the event word for the intermediary event C.
  - `\"Original Description Sentence C\"`: A string that represents the original descriptive sentence for the intermediary event C.
Note: Ensure that the Event C Event Word appears in the Original Description Sentence C, and that the content of Sentence C reflects the considerations from the previous analysis of the weak causal relationship.
- **Input**:
    - Prior causal relationship analysis: \"%s\"
    - Event A Event Word: \"%s\"
    - Original Descriptive Sentence A: \"%s\"
    - Event B Event Word: \"%s\"
    - Original Descriptive Sentence B: \"%s\""""

def generate_prompts(sample):
    edge_num = len(sample['edge'])
    node_num = len(sample['node'])

    sample_inputs = []
    prompts = ''
    for i in range(edge_num-1):
        node_c_idx = sample['edge'][i][0]
        node_c = sample['node'][node_c_idx]
        node_e_idx = sample['edge'][i][-1]
        node_e = sample['node'][node_e_idx]
        prompt = generate_template % (node_c[5], node_c[6], node_e[5], node_e[6])
        sample_inputs.append((node_c_idx, node_e_idx, node_num))
        prompts += prompt+'\n--event pair--\n'
    return sample_inputs, prompts

def generate_assessment_prompts(sample):
    edge_num = len(sample['edge'])
    node_num = len(sample['node'])

    sample_inputs = []
    prompts = ''
    for i in range(edge_num-1):
        node_c_idx = sample['edge'][i][0]
        node_c = sample['node'][node_c_idx]
        node_e_idx = sample['edge'][i][-1]
        node_e = sample['node'][node_e_idx]
        prompt = assessment_template % (node_c[5], node_c[6], node_e[5], node_e[6])
        sample_inputs.append((node_c_idx, node_e_idx, node_num))
        prompts += prompt+'\n--event pair--\n'
    return sample_inputs, prompts

def generate_enhanced_prompts_with_analysis(sample, assessment_results, sample_start_idx):
    edge_num = len(sample['edge'])
    node_num = len(sample['node'])
    
    sample_inputs = []
    prompts = ''
    
    for i in range(edge_num-1):
        node_c_idx = sample['edge'][i][0]
        node_c = sample['node'][node_c_idx]
        node_e_idx = sample['edge'][i][-1]
        node_e = sample['node'][node_e_idx]

        current_idx = sample_start_idx + i
        if current_idx < len(assessment_results):
            analysis = assessment_results[current_idx][1]
        else:
            analysis = "No analysis available"

        prompt = generate_template % (analysis, node_c[5], node_c[6], node_e[5], node_e[6])
        sample_inputs.append((node_c_idx, node_e_idx, node_num))
        prompts += prompt+'\n--event pair--\n'
    
    return sample_inputs, prompts

def pre_generate_prompts(args, datapath):
    train_data, dev_data, test_data = load_data(args)
    print('Data loaded')

    train_samples_input = []
    dev_samples_input = []
    test_samples_input = []
    with open(f'{datapath}/enhance_train_prompts.txt', 'a', encoding='utf-8') as f:
        for sample in train_data:
            sample_input, prompts = generate_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            train_samples_input.append(sample_input)
            f.write(prompts)
    with open(f'{datapath}/enhance_dev_prompts.txt', 'a', encoding='utf-8') as f:
        for sample in dev_data:
            sample_input, prompts = generate_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            dev_samples_input.append(sample_input)
            f.write(prompts)
    with open(f'{datapath}/enhance_test_prompts.txt', 'a', encoding='utf-8') as f:
        for sample in test_data:
            sample_input, prompts = generate_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            test_samples_input.append(sample_input)
            f.write(prompts)
    samples_input = {'train': train_samples_input,
                     'dev': dev_samples_input,
                     'test': test_samples_input}
    np.save(f'{datapath}/enhance_samples_input.npy', samples_input)
    print('Data enhanced')

def pre_generate_assessment_prompts(args, datapath):
    train_data, dev_data, test_data = load_data(args)
    dev_data = dev_data[:10]
    print('Assessment prompts generation started')

    with open(f'{datapath}/enhance_train_assessment_prompts.txt', 'w', encoding='utf-8') as f:
        for sample in train_data:
            sample_input, prompts = generate_assessment_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    with open(f'{datapath}/enhance_dev_assessment_prompts.txt', 'w', encoding='utf-8') as f:
        for sample in dev_data:
            sample_input, prompts = generate_assessment_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    with open(f'{datapath}/enhance_test_assessment_prompts.txt', 'w', encoding='utf-8') as f:
        for sample in test_data:
            sample_input, prompts = generate_assessment_prompts(sample)
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    print('Assessment prompts generated')

def pre_generate_enhanced_prompts_with_analysis(args, datapath, assessment_results):
    train_data, dev_data, test_data = load_data(args)
    dev_data = dev_data[:10]
    print('Enhanced prompts with analysis generation started')

    train_start_idx = 0
    dev_start_idx = sum(len(sample['edge']) - 1 for sample in train_data)
    test_start_idx = dev_start_idx + sum(len(sample['edge']) - 1 for sample in dev_data)
    
    with open(f'{datapath}/enhance_train_enhanced_prompts.txt', 'w', encoding='utf-8') as f:
        for i, sample in enumerate(train_data):
            sample_input, prompts = generate_enhanced_prompts_with_analysis(
                sample, assessment_results, train_start_idx + sum(len(s['edge']) - 1 for s in train_data[:i])
            )
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    
    with open(f'{datapath}/enhance_dev_enhanced_prompts.txt', 'w', encoding='utf-8') as f:
        for i, sample in enumerate(dev_data):
            sample_input, prompts = generate_enhanced_prompts_with_analysis(
                sample, assessment_results, dev_start_idx + sum(len(s['edge']) - 1 for s in dev_data[:i])
            )
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    
    with open(f'{datapath}/enhance_test_enhanced_prompts.txt', 'w', encoding='utf-8') as f:
        for i, sample in enumerate(test_data):
            sample_input, prompts = generate_enhanced_prompts_with_analysis(
                sample, assessment_results, test_start_idx + sum(len(s['edge']) - 1 for s in test_data[:i])
            )
            prompts = prompts + '\n--sample--\n'
            f.write(prompts)
    print('Enhanced prompts with analysis generated')

def batch_chatting(prompts, batch_size=4):
    all_results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Chatting Progress", unit="batch") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            outs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results = [out.split("assistant\n\n")[1] for out in outs]
            all_results.extend(results)

            pbar.update(1)
    return all_results

def extract_assessment_result(text):
    try:
        json_start = -1
        json_end = -1
        brace_count = 0
        start_found = False
        
        for i, char in enumerate(text):
            if char == '{':
                if not start_found:
                    json_start = i
                    start_found = True
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if start_found and brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            result = json.loads(json_str)
            need_intermediate = result.get('need_intermediate_event', False)
            analysis = result.get('analysis', '')
            return need_intermediate, analysis

        text_lower = text.lower()
        
        # 检查是否包含明确的"true"或"false"
        if '\"need_intermediate_event\": true' in text_lower or '\'need_intermediate_event\': true' in text_lower:
            return True, 'Extracted from text analysis'
        elif '\"need_intermediate_event\": false' in text_lower or '\'need_intermediate_event\': false' in text_lower:
            return False, 'Extracted from text analysis'
            
    except json.JSONDecodeError as e:
        text_lower = text.lower()

        if '\"need_intermediate_event\": true' in text_lower or '\'need_intermediate_event\': true' in text_lower:
            return True, 'Extracted from text analysis (JSON parse failed)'
        elif '\"need_intermediate_event\": false' in text_lower or '\'need_intermediate_event\': false' in text_lower:
            return False, 'Extracted from text analysis (JSON parse failed)'
            
    except Exception as e:
        return False, f'Error: {str(e)}'

def extract_event_c(text, idx, event_word_pattern, description_pattern):
    event_word_match = event_word_pattern.search(text)
    description_match = description_pattern.search(text)

    if event_word_match and description_match:
        event_word = event_word_match.group(1).strip()
        description = description_match.group(1).strip()
        if event_word in description:
            return event_word, description
        else:
            return str(idx), "not_in"
    else:
        return str(idx), "format"

def build_mapping(lst):
    mapping = {}
    count = 0
    for i, value in enumerate(lst):
        if value:
            mapping[count] = i
            count += 1
    return mapping

if __name__ == '__main__':
    args = parse_args()
    args.graph_enhance = 'None'
    dataset_type = args.dataset_type
    datapath = "enhance_graph"
    model_id = "LLMs"
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # 自动分配设备
    )
    model.eval()  # 启用评估模式
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 生成评估prompts
    pre_generate_assessment_prompts(args, datapath)

    # 第一步：评估是否需要生成中介事件
    print("Step 1: Assessing whether intermediate events are needed...")
    assessment_prompts = []
    samples_size = []
    
    with open(f"{datapath}/enhance_{dataset_type}_assessment_prompts.txt", "r", encoding="utf-8") as file:
        enhance_assessment_prompts = file.read()
        enhance_assessment_prompts_by_sample = enhance_assessment_prompts.split("\n--sample--\n")[:-1]

        for sample_prompts in enhance_assessment_prompts_by_sample:
            sample_prompts_by_pair = sample_prompts.split("\n--event pair--\n")[:-1]
            assessment_prompts.extend(sample_prompts_by_pair)
            samples_size.append(len(sample_prompts_by_pair))
        
        formatted_assessment_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in assessment_prompts
        ]

    # 批量生成评估响应
    assessment_outputs = batch_chatting(formatted_assessment_prompts, batch_size=64)
    
    # 解析评估结果，确定哪些需要生成中介事件
    need_generate_indices = []
    assessment_results = []
    
    for i, output in enumerate(assessment_outputs):
        need_intermediate, analysis = extract_assessment_result(output)
        assessment_results.append((need_intermediate, analysis))
        if need_intermediate:
            need_generate_indices.append(i)
    
    print(f"Total event pairs: {len(assessment_results)}")
    print(f"Event pairs needing intermediate events: {len(need_generate_indices)}")

    assessment_data = {
        "assessment_results": assessment_results,
        "need_generate_indices": need_generate_indices
    }
    np.save(f"{datapath}/enhance_{dataset_type}_assessment_results.npy", assessment_data)
    print("Assessment results saved")

    print("Step 2: Generating enhanced prompts with assessment analysis...")
    pre_generate_enhanced_prompts_with_analysis(args, datapath, assessment_results)

    if need_generate_indices:
        print("Step 3: Generating intermediate events for selected pairs...")

        with open(f"{datapath}/enhance_{dataset_type}_enhanced_prompts.txt", "r", encoding="utf-8") as file:
            enhance_enhanced_prompts = file.read()
            enhance_enhanced_prompts_by_sample = enhance_enhanced_prompts.split("\n--sample--\n")[:-1]

            all_prompts = []
            for sample_prompts in enhance_enhanced_prompts_by_sample:
                sample_prompts_by_pair = sample_prompts.split("\n--event pair--\n")[:-1]
                all_prompts.extend(sample_prompts_by_pair)

        selected_prompts = [all_prompts[i] for i in need_generate_indices]
        
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in selected_prompts
        ]

        parse_cnt = 0
        need_generate_flag = [True] * len(formatted_prompts)
        all_events_C_desc = [("", "")] * len(formatted_prompts)
        
        while parse_cnt < 4:
            need_generate_prompts = [formatted_prompts[i] for i in range(len(formatted_prompts)) if need_generate_flag[i]]
            mapping = build_mapping(need_generate_flag)

            outputs = batch_chatting(need_generate_prompts, batch_size=64)

            event_word_pattern = re.compile(r"Event C Event Word[\s\S]{0,10}?\"([^\"]+)\"\s*")
            description_pattern = re.compile(r"Original Description Sentence C[\s\S]{0,10}?\"([^\"]+)\"\s*")

            for i, output in enumerate(outputs):
                event_word, description = extract_event_c(output, i, event_word_pattern, description_pattern)
                all_events_C_desc[mapping[i]] = (event_word, description)
                if description != "not_in" and description != "format":
                    need_generate_flag[mapping[i]] = False
                else:
                    if description == "not_in":
                        selected_prompts[mapping[i]] += "Note: Ensure that the Event C Event Word appears in the Original Description Sentence C."
                        formatted_prompt = tokenizer.apply_chat_template(
                                            [{"role": "user", "content": selected_prompts[mapping[i]].strip()}],
                                            tokenize=False,
                                            add_generation_prompt=True
                        )
                        formatted_prompts[mapping[i]] = formatted_prompt
                    else:
                        selected_prompts[mapping[i]] += """Note: Ensure the output strictly matches the following format:
- **Event C Event Word**: \"{Event Word}\"
- **Original Description Sentence C**: \"{Original Description Sentence}\""""
                        formatted_prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": selected_prompts[mapping[i]].strip()}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        formatted_prompts[mapping[i]] = formatted_prompt
            parse_cnt += 1
            if not any(need_generate_flag):
                break

        final_events_C_desc = [("", "")] * len(all_prompts)
        for i, idx in enumerate(need_generate_indices):
            final_events_C_desc[idx] = all_events_C_desc[i]
        
        events_result = {"all_events_C_desc": final_events_C_desc, "failure_flag": [False] * len(all_prompts)}
    else:
        print("No intermediate events need to be generated.")
        with open(f"{datapath}/enhance_{dataset_type}_enhanced_prompts.txt", "r", encoding="utf-8") as file:
            enhance_enhanced_prompts = file.read()
            enhance_enhanced_prompts_by_sample = enhance_enhanced_prompts.split("\n--sample--\n")[:-1]
            all_prompts = []
            for sample_prompts in enhance_enhanced_prompts_by_sample:
                sample_prompts_by_pair = sample_prompts.split("\n--event pair--\n")[:-1]
                all_prompts.extend(sample_prompts_by_pair)
        
        events_result = {"all_events_C_desc": [("", "")] * len(all_prompts), "failure_flag": [False] * len(all_prompts)}

    np.save(f"{datapath}/enhance_{dataset_type}_events_result.npy", events_result)
    print("Processing completed!")

