import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # define GPU id, remove if you want to use all GPUs available

import sys
sys.path
sys.path.append('..')

import gc
import csv
from pathlib import Path

import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
from medusa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
import transformers
from huggingface_hub import hf_hub_download



@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def medusa_forward(input_ids, 
                   model, 
                   tokenizer, 
                   medusa_choices, 
                   temperature, 
                   posterior_threshold, 
                   posterior_alpha, 
                   max_steps=512, 
                   aux_metrics=None):
    wall_times = {'medusa': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    
    with timed(wall_times, 'init'):
        if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = model.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=model.base_model.device
            )
        model.medusa_buffers = medusa_buffers
        model.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_medusa_mode(model)
        medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
    new_token = 0

    for idx in range(max_steps): 
        with timed(wall_times, 'medusa'):
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )
            # print(medusa_buffers)

        with timed(wall_times, 'tree'):
            # outputs is the base model output
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'posterior'):
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, posterior_threshold, posterior_alpha
                )
            # logits torch.Size([42, 5, 32000])
            # candidates torch.Size([42, 5])
            # print(f"logits {logits.shape}")
            # print(f"candidates {candidates.shape}")
            
            # record correct tokens
            if aux_metrics is not None:
                if "n_matches" in aux_metrics:
                    aux_metrics["n_matches"].append(torch.Tensor.cpu(accept_length))
                if "candidate_tokens" in aux_metrics:
                    candidate_tokens = candidates[None, best_candidate, : ]
                    # print( tokenizer.decode(
                    #         candidate_tokens[0],
                    #         spaces_between_special_tokens=False,
                    #     ))
                    aux_metrics["candidate_tokens"].append(torch.Tensor.cpu(candidate_tokens))
            # print(accept_length)
            # print(best_candidate)
            # print(aux_metrics)
            # print("******")
        
        with timed(wall_times, 'update'):
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        # print(input_len)
        # print(input_ids[0, input_len:].tolist())

    return input_ids, new_token, idx, wall_times

def get_correct_tokens(ground_truth_seq, speculative_seqs, n_matches, starting_idx=0):
    
    ground_truth_seq = np.squeeze(ground_truth_seq)

    if speculative_seqs[0].shape != speculative_seqs[-1].shape:
       speculative_seqs = speculative_seqs[:-1]
       n_matches = n_matches[:-1]

    speculative_seqs = np.squeeze(speculative_seqs)
    

    idx = starting_idx
    n_correct = 0
    n_total = 0
    for i in range(len(n_matches)):
        
        # print(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy())
        # print(speculative_seqs[i])
        # print(np.where(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy() == speculative_seqs[i], 1, 0))
        ground_truth = ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy()
        speculative = speculative_seqs[i]
        if len(ground_truth) == len(speculative):
            correct = np.where(ground_truth == speculative, 1, 0).sum()
        else:
            n = np.min([len(ground_truth), len(speculative)])
            correct = np.where(ground_truth[:n] == speculative[:n], 1, 0).sum()
        
        # print(correct)
        n_correct += correct
        idx += n_matches[i] + 1
        n_total += len(speculative_seqs[i])

    return n_correct / n_total

if __name__ == "__main__":
    NUM_MEDUSA_HEADS = 2

    main_path = "/cb/home/andrewz/ws/cerebras-research/axolotl/experiments/configs/68m-llama-2-head-decay-0_6-stage2"

    model_paths = [ main_path + f"/checkpoint-{checkpoint_n*1000}" for checkpoint_n in range(1,14)]
    # model_paths = [main_path]

    prompts = [
            "Implement a Python function to compute the Fibonacci numbers. def",
            "Who does Harry turn into a balloon? <eos>",
            "What were the major contributing factors to the fall of the Roman Empire? <eos>",
            "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts. <eos>",
            "If a train travels 120 kilometers in 2 hours, what is its average speed? <eos>",
            "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig? <eos>"
            ]

    for model_path in model_paths:
        print(f"***************{model_path}***************")
        model = MedusaModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map="auto"
        )
        print(model)
        model.to('cuda')
        tokenizer = model.get_tokenizer()

        medusa_choices = mc_sim_7b_63

        temperature = 0.
        posterior_threshold = 0.09
        posterior_alpha = 0.3

        filename = f'2heads-0.6-stage2.csv'
        with open(filename, 'a') as csvfile:
            # Create a csv writer object
            csvwriter = csv.writer(csvfile)
            
            for prompt in prompts:
                print(prompt)
                with torch.inference_mode():
                    input_ids = tokenizer([prompt]).input_ids
                    
                    aux_metrics = {"n_matches": [],
                                "candidate_tokens": []
                                }
                    
                    output_ids, new_token, idx, wall_time = medusa_forward(
                                    torch.as_tensor(input_ids).cuda(),
                                    model,
                                    tokenizer,
                                    medusa_choices,
                                    temperature,
                                    posterior_threshold,
                                    posterior_alpha,
                                    aux_metrics=aux_metrics
                                )
                    # print(output_ids)
                    output_ids = output_ids[0][len(input_ids[0]) :]
                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    # print(output)
                    print("Output length:", output_ids.size(-1))

                    n_matches = aux_metrics["n_matches"]
                    
                    acceptance = np.sum(n_matches) / len(n_matches) 
                    # correct_rate_gt = get_correct_tokens(torch.Tensor.cpu(output_ids), aux_metrics["candidate_tokens"], aux_metrics["n_matches"], 0)

                    print(f"avg. token acceptance: {acceptance}")
                    # print(f'correct_rate (GT): {correct_rate_gt}')
                    
                    # print("Compression ratio:", new_token / idx)
            
                    row = [model_path, prompt, acceptance]
                    csvwriter.writerow(tuple(row))
            
            del model
            gc.collect()