### Getting gpt-eval results 
- human vs human: we want these results to get 100% accuracy for predicting the correct author style for a given human output 
    - sample command: `python benchmark_gpt4_eval -ns 50 -nd 5 -e -s 42`
        - evaluate using seed 42, 50 samples, 5 examples per sample, and generating explanations
- human vs gpt-4: we want these results to get 100% accuracy so that a gpt-4 output (generated with 0 examples in a zero-shot setting) is not chosen over a human output given some examples 
    - sample command: `python tests/benchmark_gpt4_eval.py -hu -ns 50 -nd 5 -e -s 42` 
        - evaluate using seed 42, 50 samples, 5 examples per sample, and generating explanations

To change gpt model to use and the dataset to evaluate on, edit the `openai_model_name` and `dataset` to the desired values in the `iota/configs/config_gpt4_judge.yml` file. Only `cmcc` and `speechpref` will work for `dataset` when doing human vs human because `ccat50` doesn't contain data where multiple authors share the same prompts. 