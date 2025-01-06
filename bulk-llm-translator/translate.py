#!/usr/bin/env python3

# This script ingests a file of newline-delimited sentences and translates them one at a 
# time, using a configurable prompt, to a configurable set of three or more LLMs.
#
# Alex Balashov <abalashov@evaristesys.com>

import argparse, os, sys, asyncio, math
from openai import OpenAI
from rich.progress import Progress

# Identifiers of acceptable LLM providers, and their permitted models.
llm_providers = {
    "openai": {
        "model": "gpt-4o"
    }
}

# Defaults/globals.
target_language="French"
prompt_prefix="Translate this sentence to"
sentences=[]
parallel_runners=7

async def execute_provider_pipeline(llm_pipeline):
    global prompt_prefix, target_language, sentences, parallel_runners

    chunk_size = math.ceil(len(sentences) / parallel_runners)
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    # Make a list of lists, in order to preserve ordering.
    sentences_out: list[ list[str] ] = [[] for i in range(0, len(chunks))]

    print(f"% Split {len(sentences)} sentences into {len(chunks)} chunks, where every chunk is <= {len(chunks[0])} sentences")

    for provider in llm_pipeline:
        out_file = f"out_{provider}_{llm_providers[provider]['model']}-{target_language.lower()}.txt"
        print(f"% Executing {provider}-{llm_providers[provider]['model']} pipeline, output to {out_file}:")

        async_tasks = [] 

        if provider == "openai":
            with Progress() as progress:
                progress_bars = [progress.add_task(f"Runner {i}", total=len(chunks[i])) for i in range(0, len(chunks))]

                async_tasks: list[asyncio.Task] = [
                    asyncio.to_thread(openai_task_runner, i, chunks[i], progress, progress_bars[i], sentences_out) 
                    for i in range(0, len(chunks))
                ]

                print(f"% Starting {len(async_tasks)} async tasks for OpenAI")
                await asyncio.gather(*async_tasks)
                print(f"% Finished OpenAI tasks with {len(sentences_out)} sentences translated")

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
            model="gpt-4o",
            temperature=0.0
        )

        if len(resp.choices) > 0:
            out_sentences[idx].append(resp.choices[0].message.content)
            requests_serviced = requests_serviced + 1
            progress_mgr.update(progress_bar, advance=1)

    print(f"% OpenAI task {idx} completed with {requests_serviced} sentences translated")

# Entry point.
async def main():
    global prompt_prefix, target_language, sentences, parallel_runners
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

    await execute_provider_pipeline(llm_pipeline)

if __name__ == "__main__":
    asyncio.run(main())
