import logging
import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
from peft import PeftModel

# ── EDIT THESE ────────────────────────────────────────────────────────────────
HF_NAMESPACE = "TheAIchemist13"
MODEL_REPO   = f"{HF_NAMESPACE}/lora_llama-2-7b-chat-hf-peft-unlearn-tofu"
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models"""
    seed_everything(cfg.trainer.args.seed)
    mode         = cfg.get("mode", "train")
    model_cfg    = cfg.model
    template_args= model_cfg.template_args

    # ── Build model & tokenizer ───────────────────────────────────────────────
    model, tokenizer = get_model(model_cfg)

    # ── Load data ──────────────────────────────────────────────────────────────
    data = get_data(
        cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args
    )
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    # ── Setup evaluator(s) ────────────────────────────────────────────────────
    evaluators = None
    if cfg.get("eval", None):
        evaluators = get_evaluators(
            eval_cfgs     = cfg.eval,
            template_args = template_args,
            model         = model,
            tokenizer     = tokenizer,
        )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer, targs = load_trainer(
        trainer_cfg   = cfg.trainer,
        model         = model,
        train_dataset = data.get("train", None),
        eval_dataset  = data.get("eval", None),
        tokenizer     = tokenizer,
        data_collator = collator,
        evaluators    = evaluators,
        template_args = template_args,
    )

    # ── TRAIN → MERGE & PUSH ───────────────────────────────────────────────────
    if targs.do_train:
        logger.info("Starting training…")
        trainer.train()

        # if we've wrapped with PEFT, merge adapter weights back into base
        if isinstance(model, PeftModel):
            logger.info("Merging LoRA adapters into base model weights")
            model = model.merge_and_unload()

        logger.info("Pushing model to HF Hub: %s", MODEL_REPO)
        model.push_to_hub(
            repo_id=MODEL_REPO,
            use_temp_dir=True,
            commit_message="🚀 push merged LoRA model",
            private=False,
        )

        logger.info("Pushing tokenizer to HF Hub: %s", MODEL_REPO)
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
