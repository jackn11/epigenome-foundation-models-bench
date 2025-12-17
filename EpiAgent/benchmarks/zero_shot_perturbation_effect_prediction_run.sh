# Liscovitch_Brauer2021 dataset (GSE161002)
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161002
# python zero_shot_perturbation_effect_prediction.py \
#     --csv_path /scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/Liscovitch_Brauer2021/All_data_K562_1.csv \
#     --genes_of_interest CHD5 KDM6A DNMT3A HDAC9 PBRM1 MBD1 PRDM9 ING1 EZH2 TET2 ARID1A SETD2 HIST1H3B PHF6 ATRX H3F3B SMARCB1 SMARCA4 CHD8 H3F3A CHD4 \
#     --output_dir zero_shot_perturbation_effect_prediction_Liscovitch_Brauer2021 \
#     --cache_dir ../data/embedding_cache/epiagent_embedding_cache_Liscovitch_Brauer2021\
#     --pretrained_model_path ../model/pretrained_EpiAgent.pth


# Pierce2021 dataset (GSE168851)
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168851
python zero_shot_perturbation_effect_prediction.py \
    --csv_path /scratch/wkim/project-2-team-1/EpiAgent/data/sample/genetic_perturbation_data/Pierce2021/All_data_SpearATAC_K562_LargeScreen.csv \
    --genes_of_interest sgARID2 sgARID3A sgATF1 sgATF3 sgBCLAF1 sgBRF2 sgCAD sgCDC5L sgCEBPB sgCEBPZ sgCTCF sgCUX1 sgELF1 sgFOSL1 sgGABPA sgGATA1 sgGTF2B sgHINFP sgHSPA5 sgKLF1 sgKLF16 sgMAX sgMYC sgNFE2 sgNFYB sgNRF1 sgPBX2 sgPOLR1D sgREST sgRPL9 sgSETDB1 sgTBP sgTFDP1 sgTHAP1 sgTRIM28 sgYY1 sgZBTB11 sgZNF280A sgZNF407 sgZZZ3 sgsgNT \
    --output_dir zero_shot_perturbation_effect_prediction_Pierce2021 \
    --cache_dir ../data/embedding_cache/epiagent_embedding_cache_Pierce2021 \
    --pretrained_model_path ../model/pretrained_EpiAgent.pth