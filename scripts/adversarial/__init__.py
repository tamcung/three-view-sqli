"""Chapter 4 adversarial training pipeline.

Modules:
  utils              shared model loader + libinjection-based validator
  mutations          atomic textual mutation operators (encoding / ws / case / ...)
  search_attacker    genetic algorithm over mutation chains
  hotflip_attacker   gradient-guided byte-level edit attack
  llm_attacker       LLM-as-attacker (Anthropic / OpenAI-compat / echo stub)
  co_train           round-by-round adversarial fine-tuning
  eval_robustness    end-to-end robustness reporting
"""
