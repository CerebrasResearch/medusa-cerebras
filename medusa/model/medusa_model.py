import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class HydraConfig(PretrainedConfig):
    """
    Configuration class for Hydra model.

    Args:
        hydra_num_heads (int, optional): Number of heads for the Hydra layer. Default is 2.
        hydra_num_layers (int, optional): Number of Hydra layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        hydra_num_heads=2,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        grounded_heads=False,
        hidden_state_offset = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hydra_num_heads = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.base_model_name_or_path = base_model_name_or_path
        self.grounded_heads = grounded_heads
        self.hidden_state_offset = hidden_state_offset

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
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

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass

class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")




# Hydra stuff
class HydraModelABC(nn.Module):
    """The Hydra Language Model Head.

    This module creates a series of prediction heads (based on the 'hydra' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
        # base_model = None,
        hydra_num_heads=4,
        hydra_num_layers=1,
        hydra_head_arch="mlp",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        grounded_heads=False,
        hidden_state_offset=0,
        dropout_rate=0.0,
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            hydra_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            hydra_num_layers (int, optional): Number of ResBlock layers for each Hydra head. Defaults to 0.
        """
        super().__init__(config)

        # Original model setup
        # self.base_model = config._name_or_path
        self.config = config
        # self.hidden_size = base_model.lm_head.weight.shape[-1]
        # self.vocab_size = base_model.lm_head.weight.shape[0]
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        # self.orig_lm_head = base_model.lm_head
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)

        # Hydra setup
        self.hydra = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hydra_head_arch = hydra_head_arch
        self.hidden_state_offset = hidden_state_offset
        self.dropout_rate = dropout_rate
        self.grounded_heads = grounded_heads

        if self.hydra_head_arch == "mlp":
            self.hydra_head = HydraMLP(
                hydra_num_layers=self.hydra_num_layers,
                hydra_num_heads=self.hydra,
                grounded_heads=self.grounded_heads,
                input_embed_fn=self.model.embed_tokens,
                base_config=self.config,
                lm_head_init_weight=self.lm_head.weight.data
            )
            self.hydra_lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # elif self.hydra_head_arch == "prefix-mlp":
        #     self.hydra_head = HydraPrefixMLP(
        #         hydra_num_layers=self.hydra_num_layers,
        #         hydra_num_heads=self.hydra,
        #         grounded_heads=self.grounded_heads,
        #         input_embed_fn=self.base_model.model.embed_tokens,
        #         base_config=self.config,
        #         lm_head_init_weight=base_model.lm_head.weight.data,
        #         dropout_rate=self.dropout_rate,
        #     )
        #     self.hydra_lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # elif self.hydra_head_arch == "cross-attn":
        #     self.hydra_head = HydraCrossAttentionDecoderLayer(
        #         hydra_num_layers=self.hydra_num_layers,
        #         hydra_num_heads=self.hydra,
        #         grounded_heads=self.grounded_heads,
        #         input_embed_fn=self.base_model.model.embed_tokens,
        #         base_config=self.config,
        #         lm_head=self.base_model.lm_head,
        #     )
        # elif self.hydra_head_arch == "eagle-attn":
        #     self.hydra_head = EagleAttentionDecoderLayer(
        #         hydra_num_layers=self.hydra_num_layers,
        #         hydra_num_heads=self.hydra,
        #         grounded_heads=self.grounded_heads,
        #         input_embed_fn=self.base_model.model.embed_tokens,
        #         base_config=self.config,
        #         lm_head=self.base_model.lm_head,
        #     )
        else:
            raise NotImplementedError(
                f"Hydra head architecture {self.hydra_head_arch} not supported."
            )

        # Ensure hydra_head's dtype and device align with the base_model
        self.hydra_head.to(self.base_model.dtype).to(self.base_model.device)

    @property
    def base_model(self):
        return self
        
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    # TODO (ZACK): Figure out if hydra_num_heads should just be loaded from the config
    @classmethod
    def from_pretrained(
        cls,
        hydra_head_name_or_path,
        base_model=None,
        hydra_num_heads=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            hydra_head_name_or_path (str): Name or path of the Hydra head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            HydraModel: A HydraModel instance loaded from the given path.
        """
        config = AutoConfig.from_pretrained(hydra_head_name_or_path)
        return super().from_pretrained(
            hydra_head_name_or_path,
            *args,
            **kwargs,
            config=config,
        )
        # hydra_config = HydraConfig.from_pretrained(hydra_head_name_or_path)
        # if hydra_num_heads is not None:
        #     print("Overriding hydra_num_heads as:", hydra_num_heads)
        #     hydra_config.hydra_num_heads = hydra_num_heads
        # if base_model is not None:
        #     print("Overriding base_model as:", base_model)
        #     hydra_config.base_model_name_or_path = base_model
        # base_model = HydraModelLlama.from_pretrained(
        #     hydra_config.base_model_name_or_path, **kwargs
        # )

        # model = cls(
        #     base_model,
        #     hydra_config.hydra_num_heads,
        #     hydra_config.hydra_num_layers,
        #     hydra_config.hydra_head_arch,
        #     hydra_config.base_model_name_or_path,
        #     hydra_config.grounded_heads,
        #     hydra_config.hidden_state_offset,
        # )
        # # fishy here
        # hydra_head_path = os.path.join(hydra_head_name_or_path, "hydra_lm_head.pt")
        # if os.path.exists(hydra_head_path):
        #     filename = hydra_head_path
        # else:
        #     filename = hf_hub_download(hydra_head_name_or_path, "hydra_lm_head.pt")
        # hydra_head_state_dict = torch.load(filename, map_location=base_model.device)
        # model.hydra_head.load_state_dict(hydra_head_state_dict, strict=False)

        # return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        run_hydra_head=False,
        base_hidden_states=None,
        noise_alpha=0.0,
        medusa_forward=False
    ):
        """Forward pass of the HydraModel.
        # TODO, graft this onto axolotl

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Hydra heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                # **kwargs,
            )
        if base_hidden_states is not None:
            with torch.inference_mode():
                outputs = None
                if output_orig:
                    orig_logits = self.orig_lm_head(base_hidden_states)
        else:
            with torch.inference_mode():
                # Pass input through the base model
                outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_hidden_states=self.hidden_state_offset != 0,
                )

                if output_orig:
                    orig_logits = self.base_model.lm_head(outputs[0])

            # Clone the output hidden states
            if self.hidden_state_offset == 0:
                base_hidden_states = outputs[0].clone()
            else:
                base_hidden_states = outputs[1][-(self.hidden_state_offset + 1)].clone()
        
        # Hydra heads only queried in model forward during training
        if not run_hydra_head:
            assert output_orig, "Must output original predictions if not running Hydra head."
            return None, outputs, orig_logits, base_hidden_states
        
        # From NEFT-tune
        model_dim = base_hidden_states.shape[-1]
        seq_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
        denom = torch.sqrt(seq_len * model_dim)

        noise = (torch.rand_like(base_hidden_states) * 2 - 1) * noise_alpha / denom
        noise = noise.to(base_hidden_states.dtype)
        input_base_hidden_states = base_hidden_states + noise

        # if self.hydra_head_arch == "mlp":
        hydra_logits, hydra_hidden_states = self.hydra_head(
            base_hidden_states=input_base_hidden_states, input_ids=input_ids, noise=noise
        )
        # elif self.hydra_head_arch == "prefix-mlp":
        #     hydra_logits, hydra_hidden_states = self.hydra_head(
        #         base_hidden_states=base_hidden_states,
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         past_key_values=past_key_values,
        #         position_ids=position_ids,
        #         noise=noise,
        #     )
        # elif self.hydra_head_arch == "cross-attn":
        #     hydra_logits, hydra_hidden_states = self.hydra_head(
        #         input_ids=input_ids,
        #         base_hidden_states=input_base_hidden_states,
        #         forward_mode="training",
        #         base_hidden_states_position_ids=position_ids,
        #         attention_mask=attention_mask,
        #         noise=noise,
        #     )
        #     # So that they can be stacked
        #     hydra_logits = [hydra_logits]
        #     hydra_hidden_states = [hydra_hidden_states]
        # elif self.hydra_head_arch == "eagle-attn":
        #     hydra_logits, hydra_hidden_states = self.hydra_head(
        #         input_ids=input_ids,
        #         base_hidden_states=input_base_hidden_states,
        #         forward_mode="training",
        #         position_ids=position_ids,
        #         attention_mask=attention_mask,
        #     )
        #     # So that they can be stacked
        #     hydra_logits = [hydra_logits]
        #     hydra_hidden_states = [hydra_hidden_states]

        if output_orig:
            return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs, orig_logits, base_hidden_states
        return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs
    
    def hydra_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Hydra
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        hydra_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Hydra output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            hydra_choices (list, optional): A list of integers indicating the number of choices for each Hydra head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache hydra buffers (the fixed patterns for tree attention)
        if hasattr(self, "hydra_choices") and self.hydra_choices == hydra_choices:
            # Load the cached hydra buffer
            hydra_buffers = self.hydra_buffers
        else:
            # Initialize the hydra buffer
            hydra_buffers = generate_hydra_buffers(
                hydra_choices, device=self.base_model.device
            )
        self.hydra_buffers = hydra_buffers
        self.hydra_choices = hydra_choices


        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, self.hydra_head_arch)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        
        input_len = input_ids.shape[1]

        reset_hydra_mode(self)
        # Initialize tree attention mask and process prefill tokens
        hidden_states, logits = initialize_hydra(
            input_ids, self, hydra_buffers["hydra_attn_mask"], past_key_values, hydra_buffers["proposal_cross_attn_masks"]
        )
        # print("hidden states!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(hidden_states.shape)

        new_token = 0
        last_round_token = 0
        total_accept = 0
        for idx in range(max_steps):
            # Generate candidates with topk predictions from Hydra heads
            to_pass_input_ids = None
            if idx == 0:
                to_pass_input_ids = input_ids
            candidates, tree_candidates = self.hydra_head.proposal(logits, hidden_states, hydra_buffers, past_key_values, to_pass_input_ids)

            # Use tree attention to verify the candidates and get predictions
            hidden_states, logits = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                hydra_buffers["hydra_position_ids"],
                input_ids,
                hydra_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, hydra_buffers["max_accepts"]
            )
            total_accept = accept_length + total_accept

            # Update the input_ids and logits
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
                self.hydra_head_arch,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }


            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

class HydraModelLlama(HydraModelABC, KVLlamaForCausalLM):
    pass

class HydraModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return HydraModelLlama.from_pretrained(
                pretrained_model_name_or_path,

                *args,
                **kwargs,
                
            )
        # elif config.model_type == "mistral":
        #     return MedusaModelMistral.from_pretrained(
        #         pretrained_model_name_or_path,
        #         *args,
        #         **kwargs,
        #     )
        else:
            raise ValueError("Only support llama and mistral for now!!")



class HydraMLP(nn.Module):
    """
    A MLP module as the Hydra head.

    Args:
        hidden_size (int): The size of the hidden layers in the MLP.
        num_layers (int): The number of hidden layers in the MLP.
    """

    def __init__(
        self,
        hydra_num_layers, 
        hydra_num_heads, 
        grounded_heads, 
        input_embed_fn,
        base_config,
        lm_head_init_weight=None,
    ):
        super().__init__()

        self.hidden_size = base_config.hidden_size
        self.vocab_size = base_config.vocab_size
        
        self.hydra_num_layers = hydra_num_layers
        self.hydra_num_heads = hydra_num_heads
        self.grounded_heads = grounded_heads
        self.input_embed_fn = input_embed_fn

        assert self.hydra_num_layers > 0, "Hydra MLP must have at least one layer."

        if grounded_heads:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    ResBlock(self.hidden_size, hydra_head_idx + 1),
                    *([ResBlock(self.hidden_size)] * (self.hydra_num_layers - 1))
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        else:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * self.hydra_num_layers)
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        
        self.hydra_lm_head = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size) for _ in range(self.hydra_num_heads)
        ])
        if lm_head_init_weight is not None:
            print("Initializing HydraLM head with pretrained weights...")
            for i in range(hydra_num_heads):
            # Initialize the weights of each hydra_head using the base model's weights
                self.hydra_lm_head[i].weight.data[:] = lm_head_init_weight[:]

    def forward(self, base_hidden_states, input_ids=None, noise=None):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the MLP.
        """

        hydra_hidden_states = []
        if self.grounded_heads:
            assert input_ids is not None, "Input ids must be provided for grounded heads"
            with torch.inference_mode():
                input_embeds = self.input_embed_fn(input_ids)
            if noise is not None:
                input_embeds = input_embeds + noise
            hydra_inputs = [base_hidden_states]
            for i in range(self.hydra_num_heads):
                # Move input embeddings back one spot for each hydra head idx
                hydra_inputs.append(torch.roll(input_embeds, shifts=-(i+1), dims=1))
            
            for i in range(self.hydra_num_heads):
                head_input = torch.cat(hydra_inputs[:i + 2], dim=-1) 
                hydra_hidden_states.append(self.hydra_mlp[i](head_input))
        else:
            for i in range(self.hydra_num_heads):
                hydra_hidden_states.append(self.hydra_mlp[i](base_hidden_states))
        
        hydra_logits = []
        for i in range(self.hydra_num_heads):
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_states[i]))
        
        return hydra_logits, hydra_hidden_states

    def _ungrounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        hydra_logits = []
        print(self.hydra_num_heads)
        for i in range(self.hydra_num_heads):
            # TODO: remove
            print(base_hidden_states.shape)
            hydra_hidden_state = self.hydra_mlp[i](base_hidden_states)
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_state))
        hydra_logits = torch.stack(hydra_logits, dim=0)

        # Greedy decoding: Select the most probable candidate from the original logits.
        candidates_logit = torch.argmax(input_logits[:, -1]).unsqueeze(0)

        # Extract the TOPK candidates from the hydra logits.
        candidates_hydra_logits = []
        print(hydra_buffers["beam_sizes"])
        for hydra_head, beam_size in enumerate(hydra_buffers["beam_sizes"]):
            candidates_hydra_logits.append(torch.topk(hydra_logits[hydra_head, 0, -1], beam_size, dim = -1).indices)
        candidates_hydra_logits = torch.cat(candidates_hydra_logits)

        # Combine the selected candidate from the original logits with the topk hydra logits.
        candidates = torch.cat([candidates_logit, candidates_hydra_logits.view(-1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[hydra_buffers["tree_indices"]]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[hydra_buffers["retrieve_indices"]]

        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, tree_candidates
    
    def _grounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        children_per_head = hydra_buffers["children_per_head"]
        children_to_expand_per_head = hydra_buffers["children_to_expand_per_head"]
        retrieve_indices = hydra_buffers["retrieve_indices"]

        candidate_id = torch.argmax(input_logits[:, -1]).unsqueeze(0)
        candidate_embedding = self.input_embed_fn(candidate_id).unsqueeze(0)

        candidates = torch.tensor([candidate_id], device=candidate_id.device)[None, ...]
        candidates_embeddings = torch.cat([base_hidden_states[:, -1:], candidate_embedding], dim=-1)

        for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):
            hydra_hidden_state = self.hydra_mlp[head_idx](candidates_embeddings)
            hydra_preds = self.hydra_lm_head[head_idx](hydra_hidden_state)
            next_head_embeddings = []

            for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):

                hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
                candidates = torch.cat([candidates, hydra_candidates], dim=-1)
                
                if children_to_expand > 0:
                    children_embeddings = self.input_embed_fn(hydra_candidates)[:, :children_to_expand]
                    repeat_slice = [path_idx] * children_to_expand
                    path_embeddings = candidates_embeddings[:, repeat_slice]
                    next_head_embeddings.append(torch.cat([path_embeddings, children_embeddings], dim=-1))
            
            if len(next_head_embeddings):
                # TODO (Zack): Determine assertion error about next_head_embeddings being empty before finishing tree
                candidates_embeddings = torch.cat(next_head_embeddings, dim=1)

        # TODO (Zack): Only selecting first batch element for now, change when doing bs > 1
        cart_candidates = candidates[0, retrieve_indices]

        return cart_candidates, candidates
    
    def proposal(
            self,
            input_logits,
            base_hidden_states,
            hydra_buffers,
            past_key_values=None, # Not actually used but consistent with other proposal functions,
            input_ids = None
        ):
        if self.grounded_heads:
            return self._grounded_proposal(input_logits, base_hidden_states, hydra_buffers)
        else:
            return self._ungrounded_proposal(input_logits, base_hidden_states, hydra_buffers)