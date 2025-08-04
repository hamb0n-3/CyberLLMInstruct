# LoRA Target Modules Guide

## Standard Target Modules

For consistency across different model architectures, we use a comprehensive list of target modules that covers common patterns:

```yaml
target_modules:
  # Attention modules (will match any that exist)
  - "q_proj"
  - "k_proj" 
  - "v_proj"
  - "o_proj"
  # MLP/FFN modules
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

## Model-Specific Patterns

Different models use different naming conventions:

### Qwen Models
- Uses: `self_attn.q_proj`, `self_attn.v_proj`, etc.
- MLX-LM will match these with the base names above

### GLM Models  
- Uses: `self_attn.q_proj`, `self_attn.k_proj`, etc.
- Similar pattern to Qwen

### LLaMA Models
- Uses: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`
- Also has: `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`

## Best Practice

Use the comprehensive list in all configs. MLX-LM's linear_to_lora_layers function will:
1. Search for modules containing these patterns
2. Only apply LoRA to modules that actually exist
3. Skip modules that don't match the model architecture

This ensures configs work across different model types without modification.