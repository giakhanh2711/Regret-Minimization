import logging
import config
import utils
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from dotenv import load_dotenv
import re
import getpass
import json
import itertools
import time
import pandas as pd
import numpy as np
from pathlib import Path
import os
from benchmark_experiments import Experiment


logging.getLogger("httpx").disabled = True

# Script that runs the LLM matchups and saves raw results.
class LLMExperiment:
    def __init__(self, payoffs_matrix, strategy_prompts, game_name = "", game_strategies = [], game_actions = [],
                 api_url: str = config.TAMU_BASE_URL,
                 model_name: str = config.SONNET_MODEL,
                 n_reps = config.N_REPS):

        self.game_name = game_name
        self.payoffs_matrix = payoffs_matrix
        self.strategy_prompts = strategy_prompts
        self.game_strategies = game_strategies
        self.game_actions_pattern = rf"\b({'|'.join(game_actions)})\b"
        self.api_url = api_url
        self.model = model_name
        self.n_reps = n_reps

        self.client = self.make_client()
        self.logger = utils.create_info_logger(f"{game_name}.log")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    def make_client(self):
        load_dotenv()
        key = os.environ.get("TAMU_API_KEY") or getpass.getpass("TAMU_API_KEY: ")
        raw = os.environ.get("CF_COOKIE") or getpass.getpass("CF_COOKIE: ")
        return OpenAI(api_key=key,
                    base_url=self.api_url,
                    default_headers={"Cookie": utils.normalize_cf_cookie(raw)})


    # 2.1 Parse each LLM's response to extract its chosen action
    def _visible_content(self, response):
        """Read TAMU raw text/SSE replies or an OpenAI-compatible completion.
        Return the response and model used
        """
        model = None
        if isinstance(response, str):
            if "Cloudflare Access" in response and "Sign in" in response:
                raise RuntimeError(
                    "TAMU gateway returned the Cloudflare sign-in page; refresh CF_Authorization."
                )
            if response.lstrip().startswith("data:"):
                pieces = []
                for line in response.splitlines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not model:
                        model = event["model"]
                    for choice in event.get("choices", []):
                        piece = choice.get("delta", {}).get("content")
                        if isinstance(piece, str):
                            pieces.append(piece)
                if pieces:
                    return re.sub(r"<think>.*?</think>", "", "".join(pieces), flags=re.DOTALL).strip(), model
            return response
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise TypeError(f"Unsupported TAMU completion response: {type(response).__name__}") from exc


    def _response_usage(self, response):
        if isinstance(response, str):
            if response.lstrip().startswith("data:"):
                for line in response.splitlines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage")
                    if usage is not None:
                        return usage
            else:
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    return None
        if isinstance(response, dict):
            return response.get("usage")
        return getattr(response, "usage", None)


    def choose_action(self, strategy, player_label):
        messages = [
            {"role": "system", "content": self.strategy_prompts[strategy]},
            {"role": "user", "content": f" This is the payoffs matrix {self.payoffs_matrix}. You are {player_label}. Choose your action now."},
        ]
        for attempt in range(4):
            try:
                api_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0,
                    max_tokens=16_384,
                )
                response, model = self._visible_content(api_response)
                visible = re.sub(
                    r"<think>.*?</think>", "", response,
                    flags=re.DOTALL,
                ).strip()
                actions = re.findall(self.game_actions_pattern, visible.upper())
                if not actions:
                    # Could not parse model's response
                    action = None
                else:
                    action = actions[-1]
                usage = self._response_usage(api_response)
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    completion_tokens = usage.get("completion_tokens", 0) or 0
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                return action, visible, (
                    prompt_tokens,
                    completion_tokens,
                    ), model
            
            except (RateLimitError, APIConnectionError, APIError, ValueError) as exc:
                if attempt == 3:
                    raise
                delay = 2 ** attempt
                print(f"Transient {type(exc).__name__}; retrying in {delay}s...")
                time.sleep(delay)


    def run_experiment(self):
        self.logger.info(f"N_REPS = {self.n_reps}")
        n_api_call = self.n_reps * len(self.strategy_prompts)**2 * 2
        self.logger.info(f"Total of #API calls: {n_api_call}")
        records = []
        prompt_tokens = completion_tokens = 0
        parse_failure_count = 0

        for p1_strategy, p2_strategy in itertools.product(self.strategy_prompts, repeat=2):
            for rep in range(1, self.n_reps + 1):
                time.sleep(1)
                a1, text1, usage1, model_version = self.choose_action(p1_strategy, "Player 1")
                time.sleep(1)
                a2, text2, usage2, model_version = self.choose_action(p2_strategy, "Player 2")
                if not a1 or not a2:
                    parse_failure_count += 1
                    continue

                u1, u2 = self.payoffs_matrix[(a1, a2)]
                prompt_tokens += usage1[0] + usage2[0]
                completion_tokens += usage1[1] + usage2[1]
                record = {
                    "p1_strategy": p1_strategy,
                    "p2_strategy": p2_strategy,
                    "rep": rep,
                    "p1_action": a1,
                    "p2_action": a2,
                    "p1_raw_response": text1,
                    "p2_raw_response": text2,
                    "u1": u1,
                    "u2": u2,
                    "model_version": model_version
                }
                records.append(record)
                # if raw_example is None:
                #     raw_example = {**record, "p1_visible_response": text1, "p2_visible_response": text2}
            print(f"[{p1_strategy}, {p2_strategy}]: {self.n_reps}/{self.n_reps} complete")

        output = self.format_saved_output(records)
        self.logger.info(f"Returned model: {model_version}")
        self.logger.info(f"Parse failure count {parse_failure_count}/{n_api_call}")
        cost = round(prompt_tokens / 1e9 * 3 + completion_tokens / 1e9 * 15, 5)
        self.logger.info(f"Token cost total: ${cost}")
        self.save_to_json(output)

        return records


    def format_saved_output(self, records):
        payoffs = {f"{k[0], k[1]}": list(v) for k, v in self.payoffs_matrix.items()}
        output = {
            "game": {
                "name": self.game_name,
                "actions": self.game_strategies,
                "payoffs": payoffs
            },
        
            "model": {
                "requested_model": self.model,
                "temperature": 1.0,
                "max_tokens": 16_384,
                "returned_model_version": self.model
            },
        
            "strategies": self.strategy_prompts,
        
            "trials": records
        }

        return output


    def save_to_json(self, output: dict):
        file_name = Path(f"data/{self.game_name}/{self.game_name}.json")
        file_name.parent.mkdir(parents=True, exist_ok=True)

        with open(file_name, 'w') as f:
            json.dump(output, f, indent = 2)

        self.logger.info(f"Save experiment result to {file_name}")


    def resample_payoff_matrix(self, cell_statistics: pd.DataFrame, rng):
        sampled_dict = {}
        for _, cell in cell_statistics.iterrows():
            sampled_dict[(cell.p1_strategy, cell.p2_strategy)] = (
                float(rng.normal(cell["u1_mean"], max(0.0, float(cell["u1_ci95"])) / 1.96)),
                float(rng.normal(cell["u2_mean"], max(0.0, float(cell["u1_ci95"])) / 1.96)),
            )

        df = pd.DataFrame.from_dict(sampled_dict, orient='index')
        matrix = df.to_numpy().reshape(len(self.game_strategies), len(self.game_strategies), 2)
        return sampled_dict, matrix


    def run_existing_rm_plus(self, payoffs, run_index, T=5000):
        experiment_name = f"{self.game_name}/estimation_noise/resample_{run_index}"
        experiment = Experiment(
            algo="rm+",
            payoffs=payoffs,
            game_name=experiment_name,
            T=T,
        )
        experiment.run_experiment()


    def run_estimation_noise_experiment(self, n_resamples = 10, random_seed = 2026, T = 5000):
        file_emp = Path("data") / self.game_name / f"{self.game_name}.csv"
        estimates = pd.read_csv(file_emp)

        output_dir = Path("data") / self.game_name / "estimation_noise"
        output_dir.mkdir(parents=True, exist_ok=True)

        cell_statistics = estimates

        rng = np.random.default_rng(random_seed)
        resampled_matrix_rows = []

        for run_index in range(1, n_resamples + 1):
            sampled_dict, sampled_matrix = self.resample_payoff_matrix(cell_statistics, rng)
            self.run_existing_rm_plus(sampled_matrix, run_index, T=T)
            
            for (s1, s2), (sampled_u1, sampled_u2) in sampled_dict.items():
                match = cell_statistics.loc[
                    (cell_statistics["p1_strategy"] == s1)
                    & (cell_statistics["p2_strategy"] == s2)
                ]
               
                stats = match.iloc[0]
                resampled_matrix_rows.append({
                    "resample": run_index,
                    "p1_strategy": s1,
                    "p2_strategy": s2,
                    "u1_sample_mean": stats["u1_mean"],
                    "u1_sample_se": stats["u1_ci95"],
                    "u1_resampled": sampled_u1,
                    "u2_sample_mean": stats["u2_mean"],
                    "u2_sample_se": stats["u2_ci95"],
                    "u2_resampled": sampled_u2,
                })
            print(f"Estimation-noise RM+ run {run_index}/{n_resamples} complete")
            
            pd.DataFrame(resampled_matrix_rows).to_csv(output_dir / "resampled_payoff_matrices.csv", index=False)