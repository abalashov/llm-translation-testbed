#!/usr/bin/env python3

# This script ingests a file of newline-delimited sentences and translates them one at a 
# time, using a configurable prompt, to a configurable set of three or more LLMs.
#
# Alex Balashov <abalashov@evaristesys.com>

import argparse, os, sys, asyncio, math, time 
import google.generativeai as googleai
from openai import OpenAI
from anthropic import Anthropic
from rich.progress import Progress

# Identifiers of acceptable LLM providers, and their permitted models.
llm_providers = {
    "openai": {
        "model": "gpt-4o"
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-latest"
    },
    "google": {
        "model": "gemini-1.5-pro-latest"
    }
}

# Filename suffixes based on provider and who is running it.
file_suffix_map = {
    # LLM1.a = Alex running OpenAI gpt-4o.
    "openai": "LLM.1a",
    # LLM2.a = Alex running Anthropic Claude 3.5 Sonnet.
    "anthropic": "LLM.2a",
    # LLM3.a = Alex running Gemini 1.5 Pro.
    "google": "LLM.3a"
}

# Defaults/globals.
target_language="French"
prompt_prefix="Translate this sentence to"
sentences=[]
parallel_runners=7
input_file=""

async def execute_provider_pipeline(llm_pipeline):
    global prompt_prefix, target_language, sentences, parallel_runners, input_file

    chunk_size = math.ceil(len(sentences) / parallel_runners)
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    # Make a list of lists, in order to preserve ordering.
    sentences_out: list[ list[str] ] = [[] for i in range(0, len(chunks))]

    print(f"% Split {len(sentences)} sentences into {len(chunks)} chunks, where every chunk is <= {len(chunks[0])} sentences")

    for provider in llm_pipeline:
        if provider not in llm_providers:
            print(f"Error: No file suffix map entry for provider '{provider}'")
            exit(1)

        suffix = file_suffix_map[provider]

        # Split input file by /, in case of a complex path, and take the last element.
        input_file_prox = input_file.split("/")[-1]

        # Get a current YYYYMMDDHHMMSS timestamp.
        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        out_file = f"out-[{input_file_prox}]-en-{target_language.lower()}-{current_time}-{suffix}.txt"
        print(f"% Executing {provider}-{llm_providers[provider]['model']} pipeline, output to {out_file}:")

        async_tasks = [] 

        with Progress() as progress:
            progress_bars = [progress.add_task(f"Runner {i}", total=len(chunks[i])) for i in range(0, len(chunks))]

            if provider.lower() == "openai":
                runner = openai_task_runner
            elif provider.lower() == "anthropic":
                runner = anthropic_task_runner
            elif provider.lower() == "google":
                runner = google_task_runner
            else:
                raise Exception("Unknown provider: " + provider)    

            async_tasks: list[asyncio.Task] = [
                asyncio.to_thread(runner, i, chunks[i], progress, progress_bars[i], sentences_out) 
                for i in range(0, len(chunks))
            ]

            print(f"% Starting {len(async_tasks)} async tasks for: {provider}")
            start_time = time.time()
            await asyncio.gather(*async_tasks)
            print(f"% Finished {provider} tasks in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

        # Write to file.
        with open(out_file, "w") as f:
            for block in sentences_out:
                for s in block:
                    f.write(s + "\n")

# OpenAI task runner, which prompts OpenAI to translate a chunk of sentences.
# TODO: Should be moved to a separate module for cleanliness.
def openai_task_runner(
    idx: int, 
    sentence_chunks: list[str], 
    progress_mgr: Progress,
    progress_bar: any,
    out_sentences: list[ list[str] ]
):
    global prompt_prefix, target_language
    requests_serviced: int = 0

    print(f"% Starting OpenAI task {idx} with {len(sentence_chunks)} sentences")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for s in sentence_chunks:
        resp = client.chat.completions.create(messages=[
                {
                    "role": "user",
                    "content": prompt_prefix + " " + target_language + ": " + s
                }
            ],
            model=llm_providers["openai"]["model"],
            temperature=0.0
        )

        if len(resp.choices) > 0:
            out_sentences[idx].append(resp.choices[0].message.content)
            requests_serviced = requests_serviced + 1
            progress_mgr.update(progress_bar, advance=1)

    print(f"% OpenAI task {idx} completed with {requests_serviced} sentences translated")

# Anthropic task runner, which prompts OpenAI to translate a chunk of sentences.
# TODO: Should be moved to a separate module for cleanliness.
def anthropic_task_runner(
    idx: int, 
    sentence_chunks: list[str], 
    progress_mgr: Progress,
    progress_bar: any,
    out_sentences: list[ list[str] ]
):
    global prompt_prefix, target_language, llm_providers
    requests_serviced: int = 0

    print(f"% Starting Anthropic task {idx} with {len(sentence_chunks)} sentences")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for s in sentence_chunks:
        resp = client.messages.create(messages=[
                {
                    "role": "user",
                    "content": prompt_prefix + " " + target_language + ": " + s
                }
            ],
            model=llm_providers["anthropic"]["model"],
            temperature=0.0,
            max_tokens=512
        )

        if len(resp.content) > 0:
            out_sentences[idx].append(resp.content[0].text)
            requests_serviced = requests_serviced + 1
            progress_mgr.update(progress_bar, advance=1)

        # Add some delay in order to fly under 50 RPM rate limit.
        time.sleep(1.25)

    print(f"% Anthropic task {idx} completed with {requests_serviced} sentences translated")

# Anthropic task runner, which prompts OpenAI to translate a chunk of sentences.
# TODO: Should be moved to a separate module for cleanliness.
def google_task_runner(
    idx: int, 
    sentence_chunks: list[str], 
    progress_mgr: Progress,
    progress_bar: any,
    out_sentences: list[ list[str] ]
):
    global prompt_prefix, target_language, llm_providers
    requests_serviced: int = 0

    print(f"% Starting Google task {idx} with {len(sentence_chunks)} sentences")

    googleai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = googleai.GenerativeModel(llm_providers["google"]["model"])

    for s in sentence_chunks:
        resp = model.generate_content(
            prompt_prefix + " " + target_language + ": " + s,
            generation_config=googleai.GenerationConfig(
                temperature=0.0
            )
        )

        if len(resp.candidates) == 1:
            out_sentences[idx].append(resp.candidates[0].content.parts[0].text.strip())
            requests_serviced = requests_serviced + 1
            progress_mgr.update(progress_bar, advance=1)

    print(f"% Google task {idx} completed with {requests_serviced} sentences translated")


# Entry point.
async def main():
    global prompt_prefix, target_language, sentences, parallel_runners, input_file
    llm_pipeline = []

    parser = argparse.ArgumentParser(description="Translate a file of sentences using a configurable prompt and a set of LLMs.")
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Path to the input file")
    parser.add_argument("-tl", "--target-language", required=True, type=str, help="Target language (natural name, e.g. 'French')")
    parser.add_argument("-p", "--prompt", type=str, help="LLM prompt to prefix to the translation request for every sentence, terminated by language name automatically")
    parser.add_argument("-r", "--runners", type=int, help="Number of parallel runners to use", default=10)

    parser.add_argument(
        "-prov", "--providers", 
        type=str, 
        required=True, 
        help="Comma-delimited list of LLM providers to use (e.g. 'openai,google')", 
        default="openai"
    )
    
    # Parse the arguments.
    args = parser.parse_args()

    target_language = args.target_language

    if args.runners:
        parallel_runners = args.runners

    if args.prompt:
        prompt_prefix = f"{args.prompt} {target_language}:"
        print(f"% Set custom prompt: '{prompt_prefix}'")
    else:
        prompt_prefix = f"{prompt_prefix} {target_language}:"
        print(f"% Using default prompt: '{prompt_prefix}'")

    if args.providers:
        providers = map(lambda x: x.lower(), args.providers.split(","))

        for provider in providers:
            if provider not in llm_providers:
                print(f"Error: Unknown provider '{provider}'")
                exit(1)

            # Check if {provider.upper()}_API_KEY is supplied by the environment.
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                print(f"Error: No {provider.upper()}_API_KEY key supplied from environment")
                exit(1)

            llm_pipeline.append(provider)

    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File '{args.input_file}' does not exist")
            exit(1)

        with open(args.input_file, "r") as f:
            sentences = f.readlines()

        print(f"% Ingested {len(sentences)} sentences from '{args.input_file}'")
        input_file = args.input_file

    await execute_provider_pipeline(llm_pipeline)

if __name__ == "__main__":
    asyncio.run(main())
