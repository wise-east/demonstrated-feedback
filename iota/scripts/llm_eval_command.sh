# replace mb with lb to get results 

cmcc_author_ids=2,3,5,6,7,8,11,13,15,17
ccat_author_ids=32,28,20,10,27,12,15,38,23,30

### VS author rows 

#  ditto vs author
iota.llm_eval --df1 ditto --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 ditto --df2 author -b cmcc -e -a $cmcc_author_ids -mb  # submitted
iota.llm_eval --df1 ditto --df2 author -b speechpref -e -a 0 -mb # 

# mistral 7b iota vs author
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y 
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y

iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y

# iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b speechpref -e -a 0 -mb 

# gpt4 zero vs author
iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 author -b speechpref -e -a 0 -mb 

# gpt4 fewshot vs author
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 author -b speechpref -e -a 0 -mb 

# gpt4 cot vs author
iota.llm_eval --df1 gpt-4o-0806_cot_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_cot_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted

# gpt4 opro vs author
iota.llm_eval --df1 gpt-4o-0806_opro_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_opro_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# gpt4 iota naive vs author
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted


iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted 

iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b speechpref -e -a 0 -mb 

# gpt 4 no exp vs author
iota.llm_eval --df1 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb 
iota.llm_eval --df1 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b speechpref -e -a 0 -mb 

# claude zeroshot vs author
iota.llm_eval --df1 claude-3-sonnet_zeroshot --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_zeroshot --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_zeroshot --df2 author -b speechpref -e -a 0 -mb 

# claude fewshot vs author
iota.llm_eval --df1 claude-3-sonnet_fewshot_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_fewshot_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_fewshot_ice:7 --df2 author -b speechpref -e -a 0 -mb 

# claude cot vs author
iota.llm_eval --df1 claude-3-sonnet_cot_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_cot_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted

# claude opro vs author
iota.llm_eval --df1 claude-3-sonnet_opro_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_opro_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb # submitted

# claude iota naive vs author
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y

iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y


iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b speechpref -e -a 0 -mb 

# claude iota no exp vs author
iota.llm_eval --df1 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb 
iota.llm_eval --df1 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 author -b speechpref -e -a 0 -mb 

### inter model rows

## mistral

# mistral 7b zero vs ditto
iota.llm_eval --df1 mistral7b_zeroshot --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_zeroshot --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y

# mistral 7b fewshot vs ditto 
iota.llm_eval --df1 mistral7b_fewshot_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_fewshot_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y

# mistral 7b iota vs ditto
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y 
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb -y

iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y
iota.llm_eval --df1 mistral7b_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb -y 

## gpt 4o 

# gpt4 zero vs ditto
iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # done
iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# gpt4 fewshot vs ditto
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted 
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# gpt4 cot vs ditto 
iota.llm_eval --df1 gpt-4o-0806_cot_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_cot_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted

# gpt4 opro vs ditto
iota.llm_eval --df1 gpt-4o-0806_opro_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_opro_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# gpt4 iota naive vs ditto
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted

iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted

# gpt4 iota no exp vs ditto
iota.llm_eval --df1 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb 

# gpt4 iota naive vs fewshot 
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted

iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:0_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:1_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:3_undesired:-1 --df2 gpt-4o-0806_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted 



# gpt 4 iota naive vs gpt 4 no exp
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 gpt-4o-0806_iota-no-explanations_ice:7_epochs:2_undesired:-1 -b cmcc -e -a $cmcc_author_ids -mb 

## claude 

# claude zeroshot vs ditto
iota.llm_eval --df1 claude-3-sonnet_zeroshot --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_zeroshot --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# claude fewshot vs ditto
iota.llm_eval --df1 claude-3-sonnet_fewshot_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_fewshot_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted

# claude cot vs ditto
iota.llm_eval --df1 claude-3-sonnet_cot_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_cot_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted

# claude opro vs ditto
iota.llm_eval --df1 claude-3-sonnet_opro_ice:7 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted 
iota.llm_eval --df1 claude-3-sonnet_opro_ice:7 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted

# claude iota naive vs ditto
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb # submitted

iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb # submitted  

# claude iota no exp vs ditto
iota.llm_eval --df1 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 ditto -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 --df2 ditto -b cmcc -e -a $cmcc_author_ids -mb 


# claude iota vs claude fewshot
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb -y # submitted 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb -y # submitted

iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:0_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:1_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:3_undesired:-1 --df2 claude-3-sonnet_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted


# claude iota naive vs claude iota no exp ( waiting for best iota version )
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_iota-no-explanations_ice:7_epochs:2_undesired:-1 -b cmcc -e -a $cmcc_author_ids -mb 


# cross model comparisons 

# gpt4 fewshot vs claude fewshot
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 claude-3-sonnet_fewshot_ice:7 -b ccat50 -e -a $ccat_author_ids -mb # submitted
iota.llm_eval --df1 gpt-4o-0806_fewshot_ice:7 --df2 claude-3-sonnet_fewshot_ice:7 -b cmcc -e -a $cmcc_author_ids -mb # submitted 

# gpt4 iota vs claude iota
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 -b ccat50 -e -a $ccat_author_ids -mb 
iota.llm_eval --df1 gpt-4o-0806_iota-naive_ice:7_epochs:2_undesired:-1 --df2 claude-3-sonnet_iota-naive_ice:7_epochs:2_undesired:-1 -b cmcc -e -a $cmcc_author_ids -mb 

# llm eval benchmarking 
python benchmark_llm_eval.py -ns 50 -nd 5 -e -b cmcc -mb  
python benchmark_llm_eval.py -ns 50 -nd 5 -e -b ccat50 -mb  
python benchmark_llm_eval.py -ns 50 -nd 5 -e -b ccat50 -mb -cn tfidf  

# vs human eval 
python benchmark_llm_eval.py -ns 7 -nd 5 -e -b cmcc -hu -mb  
python benchmark_llm_eval.py -ns 7 -nd 5 -e -b ccat50 -hu -mb 


# small models 
# gpt-4o mini 
iota.llm_eval --df1 gpt-4o-mini_fewshot_ice:7  --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_fewshot_ice:7  --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted

iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted

iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 gpt-4o-mini_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted 

# claude haiku 
iota.llm_eval --df1 claude-3-haiku_fewshot_ice:7 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_fewshot_ice:7 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted

iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b ccat50 -e -a $ccat_author_ids -mb -y # submitted

iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:0_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:1_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:2_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted
iota.llm_eval --df1 claude-3-haiku_iota-naive_ice:7_epochs:3_undesired:-1 --df2 author -b cmcc -e -a $cmcc_author_ids -mb -y # submitted 

