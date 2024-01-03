# llama2-finetune-finance-alpaca-colab
Fine tuning a LLaMA 2 model on Finance Alpaca using 4/8 bit quantization, easily feasible on Colab.

You will need access to LLaMA-2 via HuggingFace, replace <YOUR API TOKEN> with your Access Token from HuggingFace.

This code easily incorporates quantization to do the training on a limited infra, Tesla T4(Free Colab) would work easily with 4 bit quantization, even 8 bit depending on the input context length. 

This can be easily generalized to any other dataset, or any other model architecture apart from LLaMA(anything on HuggingFace Hub).
