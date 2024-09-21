# Description: This script contains the commands to run IOTA on the GREASE dataset.

# author_ids=(2 3 5 6 7 8 11 13 15 17)
# ccat_author_ids=(32 28 20 10 27 12 15 38 23 30)
cmcc_author_ids=2,3,5,6,7,8,11,13,15,17
ccat_author_ids=32,28,20,10,27,12,15,38,23,30
speechpref_author_ids=0

./run_iota.sh gpt-4o-0806 zeroshot ccat50 $ccat_author_ids # done
./run_iota.sh gpt-4o-0806 fewshot ccat50 $ccat_author_ids # done
./run_iota.sh gpt-4o-0806 cot ccat50 $ccat_author_ids # done
./run_iota.sh gpt-4o-0806 opro ccat50 $ccat_author_ids # done 
./run_iota.sh gpt-4o-0806 zeroshot cmcc $cmcc_author_ids # done
./run_iota.sh gpt-4o-0806 fewshot cmcc $cmcc_author_ids # done
./run_iota.sh gpt-4o-0806 cot cmcc $cmcc_author_ids # done
./run_iota.sh gpt-4o-0806 opro cmcc $cmcc_author_ids # done 
# ./run_iota.sh gpt-4o-0806 zeroshot speechpref 0
# ./run_iota.sh gpt-4o-0806 fewshot speechpref 0

./run_iota.sh claude-3-sonnet zeroshot ccat50 $ccat_author_ids # done
./run_iota.sh claude-3-sonnet fewshot ccat50 $ccat_author_ids # done
./run_iota.sh claude-3-sonnet cot ccat50 $ccat_author_ids # done
./run_iota.sh claude-3-sonnet opro ccat50 $ccat_author_ids # done 
./run_iota.sh claude-3-sonnet zeroshot cmcc $cmcc_author_ids # done
./run_iota.sh claude-3-sonnet fewshot cmcc $cmcc_author_ids # done
./run_iota.sh claude-3-sonnet cot cmcc $cmcc_author_ids # done
./run_iota.sh claude-3-sonnet opro cmcc $cmcc_author_ids # done 
# ./run_iota.sh claude-3-sonnet zeroshot speechpref 0
# ./run_iota.sh claude-3-sonnet fewshot speechpref 0

./run_iota.sh mistral7b zeroshot ccat50 $ccat_author_ids # submitted
./run_iota.sh mistral7b fewshot ccat50 $ccat_author_ids # submitted
./run_iota.sh mistral7b cot ccat50 $ccat_author_ids # submitted
./run_iota.sh mistral7b opro ccat50 $ccat_author_ids # submitted

./run_iota.sh mistral7b zeroshot cmcc $cmcc_author_ids # submitted
./run_iota.sh mistral7b fewshot cmcc $cmcc_author_ids # submitted
./run_iota.sh mistral7b cot cmcc $cmcc_author_ids # submitted
./run_iota.sh mistral7b opro cmcc $cmcc_author_ids # submitted


./run_iota.sh gpt-4o-0806 iota-naive ccat50 $ccat_author_ids # done 
./run_iota.sh gpt-4o-0806 iota-no-explanations ccat50 $ccat_author_ids
./run_iota.sh gpt-4o-0806 iota-naive cmcc $cmcc_author_ids
./run_iota.sh gpt-4o-0806 iota-no-explanations cmcc $cmcc_author_ids

./run_iota.sh claude-3-sonnet iota-naive ccat50 $ccat_author_ids # done
./run_iota.sh claude-3-sonnet iota-no-explanations ccat50 $ccat_author_ids # done
./run_iota.sh claude-3-sonnet iota-naive cmcc $cmcc_author_ids # done
./run_iota.sh claude-3-sonnet iota-no-explanations cmcc $cmcc_author_ids # done 


# small models 
# gpt-4o mini 
./run_iota.sh gpt-4o-mini fewshot ccat50 $ccat_author_ids # submitted
./run_iota.sh gpt-4o-mini iota-naive ccat50 $ccat_author_ids # submitted

./run_iota.sh gpt-4o-mini fewshot cmcc $cmcc_author_ids  # submitted
./run_iota.sh gpt-4o-mini iota-naive cmcc $cmcc_author_ids # submitted

# claude-3-haiku 
./run_iota.sh claude-3-haiku fewshot ccat50 $ccat_author_ids # submitted
./run_iota.sh claude-3-haiku iota-naive ccat50 $ccat_author_ids # submitted

./run_iota.sh claude-3-haiku fewshot cmcc $cmcc_author_ids # submitted
./run_iota.sh claude-3-haiku iota-naive cmcc $cmcc_author_ids # submitted


# mistral7b 
./run_iota.sh mistral7b iota-naive ccat50 $ccat_author_ids # done 
./run_iota.sh mistral7b iota-naive cmcc $cmcc_author_ids # done 
./run_iota.sh mistral7b iota-no-explanations ccat50 $ccat_author_ids
./run_iota.sh mistral7b iota-no-explanations cmcc $cmcc_author_ids



# test commands for iota.run_iota other baselines 
iota.run_iota /home/ec2-user/project/GREASE-IOTA/scripts/../iota/configs/config_openai.yaml --mode iota-naive --model_name_or_path gpt-4o-mini-2024-07-18 --author_key 32 --dataset ccat50 --device_id 0 --test
iota.run_iota /home/ec2-user/project/GREASE-IOTA/scripts/../iota/configs/config_openai.yaml --mode cot --model_name_or_path gpt-4o-mini-2024-07-18 --author_key 32 --dataset ccat50 --device_id 0 --test
iota.run_iota /home/ec2-user/project/GREASE-IOTA/scripts/../iota/configs/config_openai.yaml --mode opro --model_name_or_path gpt-4o-mini-2024-07-18 --author_key 32 --dataset ccat50 --device_id 0 --test