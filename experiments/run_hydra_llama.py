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
from medusa.model.medusa_model import HydraModel, HydraMLP
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

def hydra_forward(input_ids, model, tokenizer, hydra_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # Cache hydra buffers (the fixed patterns for tree attention)
    if hasattr(model, "hydra_choices") and model.hydra_choices == hydra_choices:
        # Load the cached hydra buffer
        hydra_buffers = model.hydra_buffers
    else:
        # Initialize the hydra buffer
        hydra_buffers = generate_hydra_buffers(
            hydra_choices, device=model.base_model.device
        )
    model.hydra_buffers = hydra_buffers
    model.hydra_choices = hydra_choices

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
    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids, model, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
    )
    new_token = 0


    for idx in range(max_steps): 
        to_pass_input_ids = None
        if idx == 0:
            to_pass_input_ids = input_ids
        candidates, tree_candidates = model.hydra_head.proposal(logits, 
                                                                hidden_states, 
                                                                hydra_buffers, 
                                                                past_key_values, 
                                                                to_pass_input_ids)
        # import pdb; pdb.set_trace()
        hidden_states, logits = tree_decoding_hydra(
                model,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )
        print(f"logits {logits.shape}")
        print(f"candidates {candidates.shape}")
        # var = torch.argmax(logits, dim=-1)
        # var = var[:,:-1]
        # import pdb; pdb.set_trace()
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, 0.0
            )
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            hydra_buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            past_key_values_data,
            current_length_data,
            model.hydra_head_arch
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    return input_ids, new_token, idx


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
    # NUM_MEDUSA_HEADS = 2

    main_path = "/cb/home/andrewz/ws/cerebras-research/axolotl/experiments/configs/68m-llama-4-head-decay-0_8-HYDRA/checkpoint-1000"

    # model_paths = [ Path(main_path + f"checkpoint-{checkpoint_n*1000}") for checkpoint_n in range(1,7)]
    model_paths = [Path(main_path)]

    prompts = [
            "Implement a Python function to compute the Fibonacci numbers. def",
            "Who does Harry turn into a balloon? <eos>",
            "What were the major contributing factors to the fall of the Roman Empire? <eos>",
            "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts. <eos>",
            "If a train travels 120 kilometers in 2 hour,.s, what is its average speed? <eos>",
            "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig? <eos>"
            ]

    for model_path in model_paths:
        print(f"***************{model_path}***************")
        model = HydraModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map="auto"
        )
        # print(model)
        model.to('cuda')
        tokenizer = model.get_tokenizer()

        # medusa_choices = mc_sim_7b_63
        medusa_choices = hydra

        temperature = 0.
        posterior_threshold = 0.09
        posterior_alpha = 0.3

        filename = f'hydra.csv'
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
                    
                    output_ids, new_token, idx = hydra_forward(
                                    torch.as_tensor(input_ids).cuda(),
                                    model,
                                    tokenizer,
                                    medusa_choices,
                                    temperature,
                                    posterior_threshold,
                                    posterior_alpha,
                                    # aux_metrics=aux_metrics
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