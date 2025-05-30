# LLM Translation Testbed

## Overview

This code is intended to facilitate collaboration in evaluating the quality of LLM-generated language translation against traditional neural machine translation (NMT) outputs using premium human translations as reference.

## llm-bulk-translator

This utility reads an input file of newline-delimited English sentences, and translates them to a target language of choice, using customisable prompts against a list of LLM providers.

**Initial** setup of the Python virtual environment for this utility is required to install the various LLM providers' SDK dependencies:

```
$ cd bulk-llm-translator
$ ./install.sh
```

Once this is done, activate the Python virtual environment and run `translate.py`:

```
$ cd bulk-llm-translator
$ source our-env/bin/activate
$ (our-env) $ ./translate.py
```

### Command-line options:

A full list of command-line options can be obtained via `./translate.py -h`, but the most salient ones are:

* `-tl`, `--target-language` - the target language, in English (e.g. "French"); this is appended to the LLM prompt;

* `-p`, `--prompt` (optional) - the English sentence with which to prompt the LLM. The target language and the sentence to be translated are appended to this prompt, with the appropriate delimiters (e.g. colon). If this parameter is not supplied, a default prompt is used;

* `-i`, `--input-file` - the text file containing input sentences, one sentence per line;

* `-prov`, `--providers` - a comma-separated list of LLM providers to use (e.g. `openai`, `google`, `anthropic`);

* `-r`, `--runners` - the number of API queries to run in parallel to a given LLM provider. This defaults to 10, and should be used cautiously.

* `-b`, `--batch-size` - the number of lines to batch into a single API request. This defaults to 1, which is to say that inputs will be translated line-by-line by default. If the number of lines is > 1, the prompt should appropriately reflect this, e.g. "Please translate the following list of sentences..."

### API keys from environment variables:

In addition to the required and optional command-line arguments, it is necessary to provide an API key for every LLM provider suppplied. For example, if the `openai` provider is included in the `--providers` list, the environment variable `OPENAI_API_KEY` must be set, and the same applies for `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY` and so on.

### Example usage:

Example usage:

```
$ OPENAI_API_KEY=sk-proj-xxx python3 translate.py \
  -i test_data/input_en.txt \
  -tl Russian \
  --providers openai 
```

In this invocation, `test_data/input_en.txt` contains:

```
The system must be scalable and efficient.
Scalable systems work.
Authentication is required for data integrity.
Authentication helps.
```

This invocation results in the following output to `out_openai_gpt-4o-russian.txt`:

```
Система должна быть масштабируемой и эффективной.
Масштабируемые системы работают.
Аутентификация необходима для обеспечения целостности данных.
Аутентификация помогает.
```

### Citation

if you would like to use our data or code, please cite the associated paper:

```
@misc{balashov2025translationanalyticsfreelancersi,
      title={Translation Analytics for Freelancers: I. Introduction, Data Preparation, Baseline Evaluations}, 
      author={Yuri Balashov and Alex Balashov and Shiho Fukuda Koski},
      year={2025},
      eprint={2504.14619},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.14619}, 
}
```
