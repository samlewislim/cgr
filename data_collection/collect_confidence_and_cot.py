import os
import transformers
import datasets
import torch
import re
from dataclasses import dataclass
from tqdm import tqdm
import argparse
import yaml
import pandas as pd
import time
from transformers import LogitsProcessor, LogitsProcessorList
import math

# ---- Determinism / perf toggles ----
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True  # NVIDIA A100/H100: faster matmul with TF32
    torch.backends.cudnn.benchmark = True

# ---- Precompiled regex ----
ANSWER_RE = re.compile(
    r"Answer:\s*(?:\(\s*(.*?)\s*\.\s*\)|\s*(.*?)(?:\.|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
PROB_RE = re.compile(r"Probability:\s*([01](?:\.\d+)?)", flags=re.IGNORECASE)

# ------------------------------
#  Custom logits processor
# ------------------------------
class BudgetForcingProcessor(LogitsProcessor):
    def __init__(self, start_lengths, max_think_tokens, insertion_ids):
        """
        start_lengths: list[int] prompt lengths per batch (sum attention_mask)
        max_think_tokens: int threshold after which we inject
        insertion_ids: list[int] token ids for the instruction to inject
        """
        self.start_lengths = list(map(int, start_lengths))
        self.max_think_tokens = int(max_think_tokens)
        self.insertion_ids = insertion_ids
        # Absolute index where injection starts for each sample (None -> not yet)
        self.inject_start_pos = [None] * len(self.start_lengths)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # O(1) per step, no history scanning
        batch_size, cur_len = input_ids.shape
        ins_len = len(self.insertion_ids)
        for i in range(batch_size):
            if self.inject_start_pos[i] is None:
                gen_len = cur_len - self.start_lengths[i]
                if gen_len >= self.max_think_tokens:
                    self.inject_start_pos[i] = self.start_lengths[i] + self.max_think_tokens

            start = self.inject_start_pos[i]
            if start is not None:
                matched = cur_len - start  # how many we have attempted to place so far
                if matched < ins_len:
                    forced_id = self.insertion_ids[matched]
                    scores[i].fill_(-float("inf"))
                    scores[i, forced_id] = 0.0
        return scores

# ------------------------------
#  Model wrapper
# ------------------------------
class LLMConfidenceEstimator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else None
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        # Prefer Flash-Attn2 on NVIDIA only; force SDPA on ROCm/AMD to avoid FA errors
        is_rocm = getattr(torch.version, "hip", None) is not None
        if is_rocm:
            try:
                self.model.set_attn_implementation("sdpa")
            except Exception:
                pass
        else:
            try:
                self.model.set_attn_implementation("flash_attention_2")
            except Exception:
                try:
                    self.model.set_attn_implementation("sdpa")
                except Exception:
                    pass  # fall back silently

        # cache token ids once
        self.a_token_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.b_token_id = self.tokenizer.encode("B", add_special_tokens=False)[0]

        # convenience
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        # Harmony return token (GPT-OSS); guard if absent
        try:
            self.return_id = self.tokenizer.convert_tokens_to_ids("<|return|>")
            if isinstance(self.return_id, list):  # some tokenizers can return list
                self.return_id = self.return_id[0]
        except Exception:
            self.return_id = None

    # ---------- helpers ----------
    def _kind(self):
        name = self.model_name.lower()
        if "qwen" in name:
            return "qwen"
        if "nemotron" in name:
            return "nemotron"
        if "gpt-oss" in name or "gptoss" in name or "gpt_oss" in name:
            return "gpt-oss"
        raise ValueError("Unsupported model type")

    def _chat(self, kind, formatted_prompts, use_cot=False, reasoning_effort=None):
        """
        kind in {"qwen", "nemotron", "gpt-oss"}
        returns list[str] ready-to-tokenize prompts
        """
        if kind == "qwen":
            if use_cot:
                messages = [
                    [{"role": "user", "content": "Please reason step by step. After thinking please directly provide your answer. " + p}]
                    for p in formatted_prompts
                ]
                return [
                    self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    for m in messages
                ]
            else:
                messages = [[{"role": "user", "content": p}] for p in formatted_prompts]
                return [
                    self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    for m in messages
                ]

        if kind == "nemotron":
            think_tag = "/think" if use_cot else "/no_think"
            messages = [
                [{"role": "system", "content": think_tag}, {"role": "user", "content": p}]
                for p in formatted_prompts
            ]
            return [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]

        if kind == "gpt-oss":
            eff = reasoning_effort if reasoning_effort is not None else "low"
            messages = [
                [{"role": "system", "content": f"reasoning effort: {eff}"}, {"role": "user", "content": p}]
                for p in formatted_prompts
            ]
            return [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]

        raise ValueError("Unsupported model type")

    def _oss_prefill_final(self, user_text: str, assistant_prefix: str) -> str:
        """
        Build a Harmony conversation that ends with an assistant FINAL message, prefilled.
        Ensures the next generated token is the first answer token (no analysis).
        """
        conv  = "<|start|>user<|message|>" + user_text + "<|end|>"
        conv += "<|start|>assistant<|channel|>final<|message|>" + assistant_prefix
        return conv

    # ---------- main APIs ----------
    def generate(self, prompts, use_cot=True, reasoning_effort=None):
        formatted_prompts = [p + "Format your final answer as: 'Answer:(<your_answer>.)'." for p in prompts]
        kind = self._kind()
        inputs_txt = self._chat(kind, formatted_prompts, use_cot=use_cot, reasoning_effort=reasoning_effort)

        model_inputs = self.tokenizer(
            inputs_txt,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
        ).to(self.model.device)

        max_new_tokens = 8000
        think_token_limit = 7000

        logits_processor = LogitsProcessorList()
        if use_cot and think_token_limit and think_token_limit > 0:
            if "gpt-oss" in self.model_name.lower():
                # Switch from chain-of-thought ('analysis') to user-visible answer ('final')
                insertion_text = "<|end|><|start|>assistant<|channel|>final<|message|>Answer:("
            else:
                insertion_text = " Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"
            insertion_ids = self.tokenizer.encode(insertion_text, add_special_tokens=False)
            start_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
            logits_processor.append(BudgetForcingProcessor(start_lengths, think_token_limit, insertion_ids))

        with torch.inference_mode():
            sequences = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
            )

        gen_only = sequences[:, model_inputs["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        answer_texts, lens = [], []
        for t, ids in zip(texts, gen_only):
            if "gpt-oss" in self.model_name.lower():
                raw = self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                final_tag = "<|channel|>final<|message|>"
                end_tag = "<|end|>"
                t_for_regex = raw.split(final_tag, 1)[1] if final_tag in raw else raw
                if end_tag in t_for_regex:
                    t_for_regex = t_for_regex.split(end_tag, 1)[0]
            else:
                t_for_regex = t

            m = ANSWER_RE.search(t_for_regex)
            ans = (next((g for g in reversed(m.groups()) if g is not None), "").strip() if m else "")
            answer_texts.append(ans)
            nonpad_len = int((ids != self.pad_id).sum().item()) if self.pad_id is not None else int(ids.numel())
            lens.append(nonpad_len)
        return answer_texts, texts, lens

    def get_p_true_confidence(self, prompts):
        """
        The model is prompted to produce a direct answer prefixed with 'Answer:('.
        A short A/B follow-up ("Is this answer: (A) True (B) False") is then issued
        and the model's probability for choice A is returned as p_true.

        Args:
            prompts (List[str]): Input question strings.

        Returns:
            Tuple[List[float], List[str], List[str], List[int], Optional[torch.Tensor]]:
                p_trues: per-example probability that the model labels its answer True (0.0–1.0)
                responses: raw generated responses (prefixed with 'Answer:(')
                extracted: parsed answer strings
                lens: non-pad token lengths of generated answers
                pair_logits: logits for the (A,B) tokens used to compute p_trues
        """

        formatted = [p + "\nPlease directly answer the question. Format your answer as: 'Answer:(<your answer>.)'." for p in prompts]
        kind = self._kind()

        # ---- Build inputs to ensure next token is the 1st answer token ----
        if kind == "gpt-oss":
            # Harmony: open FINAL and prefill 'Answer: '
            inputs_txt = [self._oss_prefill_final(p, "Answer:(") for p in formatted]
        elif kind == "nemotron":
            # Render with /no_think and then append 'Answer: ' so next token is answer text
            messages = [
                [{"role": "system", "content": "/no_think"}, {"role": "user", "content": p}]
                for p in formatted
            ]
            inputs_txt = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "Answer:("
                for m in messages
            ]
        else:  # qwen
            messages = [[{"role": "user", "content": p}] for p in formatted]
            inputs_txt = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:("
                for m in messages
            ]

        model_inputs = self.tokenizer(inputs_txt, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(self.model.device)

        # Use Harmony <|return|> as EOS for GPT-OSS if available; else default
        eos_id = self.return_id if (kind == "gpt-oss" and self.return_id is not None) else self.eos_id

        with torch.inference_mode():
            seqs = self.model.generate(**model_inputs, max_new_tokens=600, eos_token_id=eos_id, pad_token_id=self.pad_id)

        gen_only = seqs[:, model_inputs["input_ids"].shape[1]:]
        raw_texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # Extract answers; responses stored with 'Answer: ' prefix for consistency
        extracted, lens, responses = [], [], []
        for t, ids in zip(raw_texts, gen_only):
            # Since we prefixed 'Answer: ' in the prompt, the generated text starts with the answer content
            # Make outputs consistent with previous logs:
            resp = "Answer:(" + t.strip()
            responses.append(resp)
            m = ANSWER_RE.search(resp)
            ans = (next((g for g in m.groups() if g is not None), "").strip() if m else "")
            extracted.append(ans)
            nonpad_len = int((ids != self.pad_id).sum().item()) if self.pad_id is not None else int(ids.numel())
            lens.append(nonpad_len)

        # ---- Build follow-up for (A)/(B) with no analysis ----
        followup_inputs = []
        for idx, prompt_text in enumerate(formatted):
            if kind == "gpt-oss":
                # Rebuild convo entirely in FINAL turns
                convo  = "<|start|>user<|message|>" + prompt_text + "<|end|>"
                convo += "<|start|>assistant<|channel|>final<|message|>" + responses[idx] + "<|end|>"
                convo += "<|start|>user<|message|>Is this answer:\n(A) True\n(B) False\n<|end|>"
                convo += "<|start|>assistant<|channel|>final<|message|>The answer is: ("
                followup_inputs.append(convo)
            elif kind == "nemotron":
                msgs = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": responses[idx]},
                    {"role": "user", "content": "Is this answer:\n(A) True\n(B) False\n"},
                ]
                rendered = self.tokenizer.apply_chat_template([msgs[0], msgs[1], msgs[2], msgs[3]],
                                                              tokenize=False, add_generation_prompt=True)
                followup_inputs.append(rendered + "The answer is: (")
            else:  # qwen
                msgs = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": responses[idx]},
                    {"role": "user", "content": "Is this answer:\n(A) True\n(B) False\n"},
                ]
                rendered = self.tokenizer.apply_chat_template([msgs[0], msgs[1], msgs[2]],
                                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
                followup_inputs.append(rendered + "The answer is: (")

        f_inputs = self.tokenizer(followup_inputs, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**f_inputs).logits[:, -1, :]
            pair_logits = logits[:, [self.a_token_id, self.b_token_id]]
            pair_probs = torch.softmax(pair_logits, dim=-1)
            p_trues = pair_probs[:, 0].float().cpu().tolist()

        return p_trues, responses, extracted, lens, pair_logits
    
    def get_margin_and_predictive_entropy_confidence(self, prompts):
        """
        Compute first-step margin, token-level predictive entropy (mean over steps),
        and sequence perplexity (exp of average NLL) for model responses.

        Returns: (margins, predictive_entropies, perplexities, texts, extracted, lens)
        """
        prompts = [
            p + "\nPlease directly answer the question. Format your response as: 'Answer:(<your_answer>.)'."
            for p in prompts
        ]
        kind = self._kind()

        if kind == "gpt-oss":
            inputs_txt = [self._oss_prefill_final(p, "Answer:(") for p in prompts]
        elif kind == "nemotron":
            messages = [
                [{"role": "system", "content": "/no_think"}, {"role": "user", "content": p}]
                for p in prompts
            ]
            inputs_txt = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "Answer:("
                for m in messages
            ]
        else:  # qwen
            messages = [[{"role": "user", "content": p}] for p in prompts]
            inputs_txt = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ) + "Answer:("
                for m in messages
            ]

        # Tokenize & choose EOS for GPT-OSS if available
        model_inputs = self.tokenizer(
            inputs_txt, return_tensors="pt", padding=True, pad_to_multiple_of=8
        ).to(self.model.device)
        eos_id = self.return_id if (kind == "gpt-oss" and self.return_id is not None) else self.eos_id

        # --- 1) Generate with per-step scores (logits) ---
        with torch.inference_mode():
            gen_out = self.model.generate(
                **model_inputs,
                max_new_tokens=600,
                return_dict_in_generate=True,
                output_scores=True,       
                eos_token_id=eos_id,
                pad_token_id=self.pad_id,
            )

        sequences = gen_out.sequences
        scores = gen_out.scores

        # --- 2) First-step margin over the full vocab (token-level) ---
        first_step_probs = torch.softmax(scores[0].float(), dim=-1)  # [B, V]
        top2 = torch.topk(first_step_probs, 2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).float().cpu().tolist()

        # Precompute per-step log-probs and probs for stability
        step_log_probs = [torch.log_softmax(s.float(), dim=-1) for s in scores]  # list of [B, V]
        step_probs = [lp.exp() for lp in step_log_probs]                          # list of [B, V]

        # Slice out the generated tokens only
        gen_only = sequences[:, model_inputs["input_ids"].shape[1]:]              # [B, L_out]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        predictive_entropies, perplexities, extracted, lens = [], [], [], []
        B = sequences.size(0)
        for i in range(B):
            ids = gen_only[i]              # generated token ids for item i
            text = texts[i]
            extracted.append(text.strip())

            # Determine effective length (truncate at first pad/eos in the generated slice)
            end = ids.size(0)
            if self.pad_id is not None:
                pad_pos = (ids == self.pad_id).nonzero(as_tuple=False)
                if pad_pos.numel() > 0:
                    end = min(end, int(pad_pos[0, 0]))
            if eos_id is not None:
                eos_pos = (ids == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    end = min(end, int(eos_pos[0, 0]))
            T = int(end)
            lens.append(T)

            if T == 0:
                predictive_entropies.append(0.0)
                perplexities.append(None)
                continue

            # --- 3a) TRUE predictive entropy (token-level), averaged over steps ---
            # H_t = - sum_v p_t(v) log p_t(v);  mean over t = 1..T
            H_sum = 0.0
            # --- 3b) Sequence NLL of realized path (for perplexity) ---
            nll_sum = 0.0

            for t in range(T):
                lp_t = step_log_probs[t][i].double()  # log-softmax row [V]
                p_t = lp_t.exp()                      # softmax row [V] (safe from lp)

                # Numerically safe entropy: -sum p * log p  (xlogy handles p==0)
                H_t_val = -torch.special.xlogy(p_t, p_t).sum().item()
                H_sum += H_t_val

                # Cross-entropy contribution of the realized token at step t
                tok_id = int(ids[t])
                nll_sum += -lp_t[tok_id].item()

            pred_entropy_mean = float(H_sum / T)
            ppl = float(math.exp(nll_sum / T))

            predictive_entropies.append(pred_entropy_mean)
            perplexities.append(ppl)

        return margins, predictive_entropies, perplexities, texts, extracted, lens

    def get_verbalised_confidence(self, prompts):
        prompts = [p + "\nPlease directly provide your best guess of the answer to the question and give the probability that you think it is correct (0.0 to 1.0). Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Give ONLY the guess and probability, no other words or explanation.\nFormat your final response as: 'Answer:(<your_best_guess>.) Probability: <score between 0.0 and 1.0>'." for p in prompts]
        kind = self._kind()

        if kind == "gpt-oss":
            inputs_txt = [self._oss_prefill_final(p, "Answer:(") for p in prompts]
        elif kind == "nemotron":
            messages = [
                [{"role": "system", "content": "/no_think"}, {"role": "user", "content": p}]
                for p in prompts
            ]
            inputs_txt = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "Answer:("
                for m in messages
            ]
        else:  # qwen
            messages = [[{"role": "user", "content": p}] for p in prompts]
            inputs_txt = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:("
                for m in messages
            ]

        model_inputs = self.tokenizer(inputs_txt, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(self.model.device)
        eos_id = self.return_id if (kind == "gpt-oss" and self.return_id is not None) else self.eos_id

        with torch.inference_mode():
            seqs = self.model.generate(**model_inputs, max_new_tokens=500, eos_token_id=eos_id, pad_token_id=self.pad_id)

        gen_only = seqs[:, model_inputs["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        confidences, extracted, lens = [], [], []
        for t, ids in zip(texts, gen_only):
            mprob = PROB_RE.search(t)
            confidences.append(float(mprob.group(1)) if mprob else None)

            m = re.search(r"Answer:\s*(?:\(\s*(.+?)\s*\.\s*\)|(.+?))\s*(?:Probability:|$)", t, flags=re.IGNORECASE | re.DOTALL)
            ans = (next((g for g in m.groups() if g is not None), "").strip() if m else t.split("Probability:")[0].strip())
            extracted.append(ans)
            nonpad_len = int((ids != self.pad_id).sum().item()) if self.pad_id is not None else int(ids.numel())
            lens.append(nonpad_len)
        return confidences, texts, extracted, lens


# ------------------------------
#  Experiment harness
# ------------------------------
@dataclass
class ExperimentalConfig:
    dataset_path: str
    model_name: str
    dataset_prefix: str = ""
    batch_size: int = 2
    resume: bool = False
    save_every: int = 200

def answer_correct(answer: str, ground_truth: str) -> bool:
    def clean(s):
        return re.sub(r"[().]", "", s).strip().lower()
    return clean(answer) == clean(ground_truth)

def main(config: ExperimentalConfig):
    dataset = datasets.load_dataset("parquet", data_files=config.dataset_path)
    model = LLMConfidenceEstimator(config.model_name)

    output_dir = "confidence_and_cot_results"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{config.model_name.replace('/', '_')}_{os.path.basename(config.dataset_path).replace('.parquet', '')}_results.parquet"
    output_path = os.path.join(output_dir, output_filename)

    existing_df = pd.DataFrame()
    if config.resume and os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)

    processed_ids = set(existing_df["question_id"].tolist()) if not existing_df.empty and "question_id" in existing_df.columns else set()

    batch_size = config.batch_size
    to_process = []
    for row_idx in range(len(dataset["train"])):
        row = dataset["train"][row_idx]
        qid = row.get("id", row_idx)
        if qid not in processed_ids:
            to_process.append((row_idx, config.dataset_prefix + row["question"]))

    print(f"Resuming: {len(processed_ids)} already processed, {len(to_process)} remaining.")

    def save_progress(new_rows: list):
        nonlocal existing_df
        if not new_rows:
            return
        new_df = pd.DataFrame(new_rows)
        combined = new_df if existing_df.empty else pd.concat([existing_df, new_df], ignore_index=True)
        if "question_id" in combined.columns:
            combined.drop_duplicates(subset=["question_id"], keep="last", inplace=True)
        tmp_path = output_path + ".tmp"
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        existing_df = combined

    pending_rows = []

    for start in tqdm(range(0, len(to_process), batch_size), desc="Processing"):
        batch_items = to_process[start:start + batch_size]
        if not batch_items:
            continue

        batch_indices = [idx for idx, _ in batch_items]
        batch_questions = [q for _, q in batch_items]

        avg_timings = {}

        t0 = time.time()
        p_true_confidences, resp_pt, ans_pt, lens_pt, logits_pt = model.get_p_true_confidence(batch_questions)
        avg_timings['p_true'] = (time.time() - t0) / len(batch_questions)

        t0 = time.time()
        v_conf, resp_v, ans_v, lens_v = model.get_verbalised_confidence(batch_questions)
        avg_timings['verbalised'] = (time.time() - t0) / len(batch_questions)

        t0 = time.time()
        margins, pes, perps, resp_pe, ans_pe, lens_pe = model.get_margin_and_predictive_entropy_confidence(batch_questions)
        avg_timings['margin_pe'] = (time.time() - t0) / len(batch_questions)

        if "gpt-oss" in model.model_name.lower():
            t0 = time.time()
            cot_answer_low, cot_resp_low, cot_len_low = model.generate(batch_questions, use_cot=True, reasoning_effort="low")
            avg_timings['cot_low'] = (time.time() - t0) / len(batch_questions)
            t0 = time.time()
            cot_ans_med, cot_resp_med, cot_len_med = model.generate(batch_questions, use_cot=True, reasoning_effort="medium")
            avg_timings['cot_medium'] = (time.time() - t0) / len(batch_questions)
            t0 = time.time()
            cot_ans_high, cot_resp_high, cot_len_high = model.generate(batch_questions, use_cot=True, reasoning_effort="high")
            avg_timings['cot_high'] = (time.time() - t0) / len(batch_questions)
        else:
            t0 = time.time()
            cot_ans, cot_resp, cot_len = model.generate(batch_questions, use_cot=True)
            avg_timings['cot'] = (time.time() - t0) / len(batch_questions)

        for i, row_idx in enumerate(batch_indices):
            row = dataset['train'][row_idx]
            gt = row.get('answer', None)

            direct_answer_correct_p_true = answer_correct(ans_pt[i], gt)
            direct_answer_correct_verbalised = answer_correct(ans_v[i], gt)
            direct_answer_correct_margin_pe = answer_correct(ans_pe[i], gt)

            base = {
                "question_id": row.get('id', row_idx),
                "question": row['question'],
                "p_true_confidence": p_true_confidences[i],
                "direct_response_p_true": resp_pt[i],
                "direct_extracted_answer_p_true": ans_pt[i],
                "direct_answer_len_p_true": lens_pt[i],
                "direct_answer_correct_p_true": direct_answer_correct_p_true,
                "direct_answer_p_true_logits": logits_pt[i].float().cpu().tolist() if logits_pt is not None else None,
                "avg_timing_p_true": avg_timings['p_true'],
                "verbalised_confidence": v_conf[i],
                "direct_response_verbalised": resp_v[i],
                "direct_extracted_answer_verbalised": ans_v[i],
                "direct_answer_len_verbalised": lens_v[i],
                "direct_answer_correct_verbalised": direct_answer_correct_verbalised,
                "avg_timing_verbalised": avg_timings['verbalised'],
                "margin_confidence": margins[i],
                "predictive_entropy_confidence": pes[i],
                "perplexity_confidence": perps[i],
                "direct_response_margin_pe": resp_pe[i],
                "direct_extracted_answer_margin_pe": ans_pe[i],
                "direct_answer_len_margin_pe": lens_pe[i],
                "direct_answer_correct_margin_pe": direct_answer_correct_margin_pe,
                "avg_timing_margin_pe": avg_timings['margin_pe'],
                "ground_truth": gt,
            }

            if "gpt-oss" in model.model_name.lower():
                base.update({
                    "high_cot_response": cot_resp_high[i],
                    "high_cot_extracted_answer": cot_ans_high[i],
                    "high_cot_answer_len": cot_len_high[i],
                    "high_cot_answer_correct": answer_correct(cot_ans_high[i], gt),
                    "avg_timing_cot_high": avg_timings['cot_high'],
                    "medium_cot_response": cot_resp_med[i],
                    "medium_cot_extracted_answer": cot_ans_med[i],
                    "medium_cot_answer_len": cot_len_med[i],
                    "medium_cot_answer_correct": answer_correct(cot_ans_med[i], gt),
                    "avg_timing_cot_medium": avg_timings['cot_medium'],
                    "low_cot_response": cot_resp_low[i],
                    "low_cot_extracted_answer": cot_answer_low[i],
                    "low_cot_answer_len": cot_len_low[i],
                    "low_cot_answer_correct": answer_correct(cot_answer_low[i], gt),
                    "avg_timing_cot_low": avg_timings['cot_low'],
                })
            else:
                base.update({
                    "cot_response": cot_resp[i],
                    "cot_extracted_answer": cot_ans[i],
                    "cot_answer_len": cot_len[i],
                    "cot_answer_correct": answer_correct(cot_ans[i], gt),
                    "avg_timing_cot": avg_timings['cot'],
                })

            pending_rows.append(base)

        if (len(pending_rows) >= config.save_every) or (start + batch_size >= len(to_process)):
            save_progress(pending_rows)
            processed_ids.update([r["question_id"] for r in pending_rows])
            pending_rows.clear()

    save_progress(pending_rows)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = ExperimentalConfig(**config_dict)
    main(config)
