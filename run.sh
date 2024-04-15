python launch.py --config configs_debug/sweetdreamer-stage1.yaml --train --gpu 0 \
                 system.prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 system.cmm_prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 tag=einstein

python launch.py --config configs/sweetdreamer-stage2.yaml --train --gpu 0 \
                 system.prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 system.cmm_prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 tag=einstein