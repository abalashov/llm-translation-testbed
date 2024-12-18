#!/usr/bin/env python3

# This script ingests a file of newline-delimited sentences and translates them one at a 
# time, using a configurable prompt, to a configurable set of three or more LLMs.
#
# Alex Balashov <abalashov@evaristesys.com>

import argparse, os, sys 
from openai import OpenAI

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

def execute_provider_pipeline(llm_pipeline):
    global prompt_prefix
    global target_language
    global sentences

    for provider in llm_pipeline:
        out_file = f"out_{provider}_{llm_providers[provider]['model']}.txt"
        print(f"% Executing {provider}-{llm_providers[provider]['model']} pipeline, output to {out_file}:")
        
        if provider == "openai":
            with open(out_file, "w") as f:
                for s in sentences:          
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    resp = client.chat.completions.create(messages=[
                            {
                                "role": "user",
                                "content": prompt_prefix + " " + target_language + ": " + s
                            }
                        ],
                        model=f"{llm_providers[provider]['model']}"
                    )

                    if len(resp.choices) > 0:
                        f.write(f"{resp.choices[0].message.content}\n")


# Entry point.
def main():
    global prompt_prefix, target_language, sentences
    llm_pipeline = []

    parser = argparse.ArgumentParser(description="Translate a file of sentences using a configurable prompt and a set of LLMs.")
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Path to the input file")
    parser.add_argument("-tl", "--target-language", required=True, type=str, help="Target language (natural name, e.g. 'French')")
    parser.add_argument("-p", "--prompt", type=str, help="LLM prompt to prefix to the translation request for every sentence, terminated by language name automatically")

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

    execute_provider_pipeline(llm_pipeline)

if __name__ == "__main__":
    main()
