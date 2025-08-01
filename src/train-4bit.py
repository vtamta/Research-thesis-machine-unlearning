import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from data import get_data, get_collators
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
from bitsandbytes.nn import Linear4bit

# HF repos
HF_NAMESPACE = "TheAIchemist13"
MODEL_REPO   = f"{HF_NAMESPACE}/unlearn-tofu_qunatize_lora"
CODE_REPO    = f"{HF_NAMESPACE}/unlearn-tofu-code"

logging.basicConfig(
    format="%(Y-%m-%d %H:%M:%S) %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

def get_model(model_cfg):
    logger = logging.getLogger(__name__)
    logger.info("=== MODEL CONFIGURATION ===\n%s", OmegaConf.to_yaml(model_cfg))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer_args.pretrained_model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            "No pad_token defined; setting pad_token to eos_token %r (id=%d)",
            tokenizer.eos_token, tokenizer.eos_token_id,
        )

    # ── 4‑bit Quant (BitsAndBytes) ────────────────────────────────────────────
    raw_bnb = getattr(model_cfg, "bnb_config", None)
    bnb_config = None
    if raw_bnb:
        bnb_dict = (
            raw_bnb
            if isinstance(raw_bnb, dict)
            else OmegaConf.to_container(raw_bnb, resolve=True)
        )
        logger.info("Enabling 4‑bit quantization: %s", bnb_dict)
        bnb_config = BitsAndBytesConfig(**bnb_dict)

    # ── Model Load ─────────────────────────────────────────────────────────────
    margs = model_cfg.model_args
    logger.info("Loading model '%s'", margs.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        margs.pretrained_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded. 4‑bit: %s", bool(bnb_config))

    # ── LoRA / PEFT ────────────────────────────────────────────────────────────
    raw_peft = getattr(model_cfg, "peft", None)
    if raw_peft:
        peft_dict = (
            raw_peft
            if isinstance(raw_peft, dict)
            else OmegaConf.to_container(raw_peft, resolve=True)
        )
        logger.info("Attaching LoRA adapters: %s", peft_dict)
        lora_cfg = LoraConfig(**peft_dict)
        model = get_peft_model(model, lora_cfg)

        total, trainable = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(
            "LoRA params: %d/%d (%.3f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    return model, tokenizer

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    logger = logging.getLogger(__name__)

    # dump model subtree
    logger.info("=== MODEL CONFIGURATION ===\n%s", OmegaConf.to_yaml(cfg.model))

    # load
    model, tokenizer = get_model(cfg.model)

    # summary
    margs = cfg.model.get("model_args", {}) or {}
    name  = margs.get("pretrained_model_name_or_path", "<unknown>")
    bnb   = cfg.model.get("bnb_config", {}) or {}
    if bnb.get("load_in_4bit", False):
        bits = bnb.get("bnb_4bit_quant_type", "?")
        dt   = bnb.get("bnb_4bit_compute_dtype", "?")
        logger.info(
            "=== MODEL SUMMARY ===\n"
            f" • name:       {name}\n"
            f" • quantized:  yes ({bits}-bit, compute={dt})"
        )
    else:
        logger.info(
            "=== MODEL SUMMARY ===\n"
            f" • name:       {name}\n"
            f" • quantized:  no (full precision)"
        )
    has4 = any(isinstance(m, Linear4bit) for m in model.modules())
    logger.info(" • 4‑bit layers present: %s", has4)

    # data + collator
    data = get_data(
        cfg.data,
        mode=cfg.get("mode", "train"),
        tokenizer=tokenizer,
        template_args=cfg.model.template_args,
    )
    logger.info("Data splits available: %s", list(data.keys()))
    if "train" in data:
        logger.info(" ➤ 'train' split size: %d", len(data["train"]))
    if "eval" not in data:
        logger.info(" ➤ No 'eval' split found (eval_dataset=None).")

    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    # optional evaluators
    evaluators = None
    if cfg.get("eval", None):
        evaluators = get_evaluators(
            eval_cfgs    = cfg.eval,
            template_args= cfg.model.template_args,
            model        = model,
            tokenizer    = tokenizer,
        )

    # trainer
    trainer, targs = load_trainer(
        trainer_cfg   = cfg.trainer,
        model         = model,
        train_dataset = data.get("train", None),
        eval_dataset  = data.get("eval", None),
        tokenizer     = tokenizer,
        data_collator = collator,
        evaluators    = evaluators,
        template_args = cfg.model.template_args,
    )

    # ── TRAIN & PUSH ─────────────────────────────────────────────────────────
    if targs.do_train:
        logger.info("Starting training…")
        trainer.train()

        # ----- MERGE LoRA → base -----
        if isinstance(model, PeftModel):
            logger.info("Merging LoRA adapters into base model weights")
            model = model.merge_and_unload()

        # ----- PUSH MERGED MODEL + TOKENIZER -----
        logger.info("Pushing merged quant+LoRA model to HF Hub: %s", MODEL_REPO)
        model.push_to_hub(
            repo_id=MODEL_REPO,
            use_temp_dir=True,
            commit_message="🚀 push merged quantized+LoRA model",
            private=False,
        )
        tokenizer.push_to_hub(
            repo_id=MODEL_REPO,
            commit_message="🚀 add tokenizer",
        )
        logger.info("✅ Push complete.")

    # ── EVAL ──────────────────────────────────────────────────────────────────
    if targs.do_eval:
        logger.info("Starting evaluation…")
        res = trainer.evaluate(metric_key_prefix="eval")
        logger.info("Eval results: %s", res)

if __name__ == "__main__":
    main()
