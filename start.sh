python generate.py  --method speculative \
                    --max_new_tokens 64 \
                    --target_model /root/model/facebook/opt-1.3b \
                    --draft_model /root/model/facebook/opt-125m \
                    --batch_size 1 \
                    --temperature 0

# python generate.py  --method autoregressive \
#                     --max_new_tokens 64 \
#                     --target_model /root/model/facebook/opt-1.3b \
#                     --batch_size 1 \
                    # --temperature 0.5

# python benchmark.py

python generate.py  --method speculative \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model /root/model/facebook/opt-1.3b \
                    --draft_model /root/model/facebook/opt-125m \
                    --temperature 0


python generate.py  --method autoregressive \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model /root/model/facebook/opt-1.3b \
                    --temperature 0